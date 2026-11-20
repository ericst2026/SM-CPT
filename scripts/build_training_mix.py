#!/usr/bin/env python3
"""Step 3 - build: turn data/split/* into training-ready files for Qwen3-1.7B, from a versioned recipe.

    python scripts/build_training_mix.py --config configs/mix/v0.2.yaml       # recommended
    python scripts/build_training_mix.py                                      # defaults (= v0.1 behaviour)
    python scripts/snapshot_data_version.py --version v0.2 --note "..."       # then freeze the result

Pipeline
  * quality filters      empty/short answers, refusals, language
  * exact dedup          normalised question text across sources (curated sources win)
  * decontamination      13-gram overlap with benchmark items (ai_eval bucket, local MMLU, frozen eval files)
  * level fix   (v0.2)   answers that do not fit their level label are relabelled (or dropped);
                         Cosmopedia questions get their audience phrase rewritten to match
  * reasoning   (v0.2)   <think> traces kept when they fit the context; R1 traces stored with RL-math
                         problems become SFT examples; over-long traces stripped or dropped
  * RL routing  (v0.2)   MCQ items with a known answer letter split between SFT and rl_mcq.jsonl
  * reasoning   (v0.2)   general (non-AI) reasoning from data/split/reasoning_sft as a fixed share of SFT,
                         round-robin across math / code / logic / science, no level system prompt
  * balancing            per-source caps (with overrides), per-level caps, replay ratio
  * formatting           chat `messages` with a level system prompt (replay: none in v0.2)

Output (--out, default data/final): sft_train/sft_val.jsonl, rl_math/rl_code/rl_mcq.jsonl, eval/*,
stats.json, stats.md, mix_config.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_qa_data.mix_rules import (PLAIN_LEVELS, fits, fix_level, is_mcq, reasoning_ok,  # noqa: E402
                                  rewrite_audience, rl_math_to_reasoning_sft)
from ai_qa_data.taxonomy import LEVELS, TOPICS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRIORITY = ["se_ai", "se_datascience", "ai_history_timeline", "ml_qa_mjphayes", "ai_qa_harikrishnan",
            "ds_qa_soufyane", "ml_qa_prsdm", "simplewiki_ai", "arxiv_ml_abstracts", "se_stats", "so_python_dsml",
            "ml_qa_collection", "cosmo_stanford", "cosmo_khanacademy", "cosmo_v2_smollm", "cosmo_web_v2"]

REFUSAL_RE = re.compile(r"^\s*(?:I'?m sorry|I am sorry|I apologi[sz]e|As an AI(?: language model)?,? I (?:can(?:no|')t|do not|don'?t)|"
                        r"I (?:can(?:no|')t|am unable to) (?:help|assist|provide))", re.I)
WORD_RE = re.compile(r"[a-z0-9]+")
MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
CODE_SUFFIX = ("\n\nWrite a complete, self-contained Python solution in a single ```python``` code block. "
               "Keep the function names and signatures given in the problem.")
MCQ_SUFFIX = "\n\nThink step by step, then give the letter of the correct option inside \\boxed{}."

# Defaults reproduce v0.1 behaviour; a --config file overrides them.
DEFAULTS = {
    "version": "unversioned", "frozen_eval": None, "exclude_from_train": [], "level_fix": "off",
    "replay_system_prompt": "general_public", "source_cap_default": 50000, "source_caps": {},
    "level_caps": {}, "level_cap_default": 80000, "replay_ratio": 0.10, "reasoning": "think",
    "max_reasoning_chars": None, "long_reasoning": "keep", "rl_math_traces_to_sft": False,
    "reasoning_source_cap": None, "mcq_rl_fraction": 0.0, "rl_math_cap": 60000, "rl_code_cap": 30000,
    "rl_mcq_cap": 30000, "val_frac": 0.02, "qa_eval_per_level": 150, "langs": ["en"], "seed": 0,
    "decontam_dirs": ["data/eval"],
    "reasoning_ratio": 0.0, "reasoning_cap_per_source": 15000,
    "reasoning_categories": ["math", "code", "logic", "science", "general"],
}


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_key(text: str) -> str:
    return hashlib.md5(" ".join(WORD_RE.findall(text.lower()))[:2000].encode()).hexdigest()


def frac_hash(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def ngrams(text: str, n: int = 13) -> set[str]:
    w = WORD_RE.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(0, max(0, len(w) - n + 1))}


def load_bucket(split_dir: str, bucket: str) -> list[dict]:
    recs = []
    for p in sorted(glob.glob(os.path.join(split_dir, bucket, "*.jsonl"))):
        recs.extend(read_jsonl(p))
    return recs


def source_rank(name: str) -> int:
    return PRIORITY.index(name) if name in PRIORITY else len(PRIORITY)


def good_sft(r: dict, langs: set[str]) -> bool:
    if r.get("lang", "en") not in langs:
        return False
    q, a = r.get("question") or "", r.get("answer") or ""
    return len(q.strip()) >= 8 and len(a.strip()) >= 20 and not REFUSAL_RE.search(a)


def assistant_content(r: dict, reasoning_mode: str) -> str:
    a = r["answer"].strip()
    if reasoning_mode == "think" and r.get("reasoning"):
        return f"<think>\n{r['reasoning'].strip()}\n</think>\n\n{a}"
    return a


def to_sft(r: dict, reasoning_mode: str) -> dict:
    level = r["level"]
    sys_msgs = [] if level == "none" else [{"role": "system", "content": LEVELS[level]["system"]}]
    if r.get("messages"):
        msgs = sys_msgs + [dict(m) for m in r["messages"] if m["role"] != "system"]
        for m in msgs:
            if m["role"] == "assistant":
                if m["content"].strip() == (r.get("answer") or "").strip():
                    m["content"] = assistant_content({**r, "answer": m["content"]}, reasoning_mode)
                break
    else:
        msgs = sys_msgs + [{"role": "user", "content": r["question"].strip()},
                           {"role": "assistant", "content": assistant_content(r, reasoning_mode)}]
    return {"id": r["id"], "messages": msgs, "level": level, "topics": r.get("topics", []),
            "source": r["source"], "has_reasoning": bool(r.get("reasoning")) and reasoning_mode == "think",
            "level_fix": r.get("level_fix")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="mix recipe YAML, e.g. configs/mix/v0.2.yaml")
    ap.add_argument("--split", default=os.path.join(ROOT, "data", "split"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "final"))
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    path = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)  # noqa: E731
    rng = random.Random(cfg["seed"])
    langs = set(cfg["langs"])
    os.makedirs(os.path.join(args.out, "eval"), exist_ok=True)
    stats: dict = {"version": cfg["version"], "dropped": Counter(), "level_fix": Counter(), "reasoning_fix": Counter()}
    print(f"building mix {cfg['version']} -> {args.out}")

    # ---------------- contamination / exclusion indexes ----------------
    evals = load_bucket(args.split, "ai_eval")
    contam = set()
    for r in evals:
        contam |= ngrams(r["question"])
    n_local = 0
    import pyarrow.parquet as pq
    for d in cfg["decontam_dirs"]:
        for f in glob.glob(os.path.join(path(d), "**", "*.parquet"), recursive=True):
            if not os.path.basename(f).startswith(("test", "validation")) or "/all/" in f:
                continue
            try:
                qs = pq.read_table(f, columns=["question"]).column("question").to_pylist()
            except Exception:  # noqa: BLE001
                continue
            for q in qs:
                contam |= ngrams(q or "")
                n_local += 1
    stats["decontam_local_questions"] = n_local
    exclude_keys, exclude_ids = set(), set()
    frozen = path(cfg["frozen_eval"]) if cfg["frozen_eval"] else None
    if frozen:
        for f in ("mcq.jsonl", "code.jsonl"):
            p = os.path.join(frozen, f)
            if os.path.exists(p):
                for r in read_jsonl(p):
                    contam |= ngrams(r["question"])
        p = os.path.join(frozen, "open_qa.jsonl")
        if os.path.exists(p):
            for r in read_jsonl(p):
                exclude_ids.add(r["id"])
                exclude_keys.add(norm_key(r["question"]))
    for p in cfg["exclude_from_train"]:
        for r in read_jsonl(path(p)):
            exclude_ids.add(r["id"])
            u = next((m["content"] for m in r.get("messages", []) if m["role"] == "user"), None)
            if u:
                exclude_keys.add(norm_key(u))
    stats["excluded_eval_val_keys"] = len(exclude_keys)

    def contaminated(r):
        return bool(contam) and not contam.isdisjoint(ngrams(r["question"]))

    # ---------------- SFT pool ----------------
    sft = load_bucket(args.split, "ai_sft")
    if cfg["rl_math_traces_to_sft"]:
        conv = [x for x in (rl_math_to_reasoning_sft(r) for r in load_bucket(args.split, "ai_rl_math")) if x]
        stats["rl_math_traces_converted"] = len(conv)
        sft += conv
    sft.sort(key=lambda r: source_rank(r["source"]))
    seen, kept = set(), []
    fit_before = Counter()
    rl_mcq = []
    for r in sft:
        if not good_sft(r, langs):
            stats["dropped"]["quality"] += 1
            continue
        k = norm_key(r["question"])
        if k in seen:
            stats["dropped"]["duplicate"] += 1
            continue
        seen.add(k)
        if r["id"] in exclude_ids or k in exclude_keys:
            stats["dropped"]["frozen_eval_or_val"] += 1
            continue
        if contaminated(r):
            stats["dropped"]["contaminated"] += 1
            continue
        # level fix
        fit_before[fits(r["level"], r["answer"])[0]] += 1
        if cfg["level_fix"] != "off":
            new, why = fix_level(r, cfg["level_fix"])
            stats["level_fix"][why.split(":")[0] if why.startswith(("relabel", "drop")) else why] += 1
            if why.startswith("relabel"):
                stats["level_fix"][why] += 1
            if new is None:
                stats["dropped"]["level_misfit"] += 1
                continue
            if new != r["level"]:
                r["level_original"], r["level"], r["level_fix"] = r["level"], new, why
                r["question"] = rewrite_audience(r["question"], new)
        # reasoning length
        if r.get("reasoning") and cfg["max_reasoning_chars"] and not reasoning_ok(r, cfg["max_reasoning_chars"]):
            if cfg["long_reasoning"] == "drop" or r["source"].endswith("_r1"):   # converted traces: the trace IS the point
                stats["reasoning_fix"]["dropped_long"] += 1
                continue
            if cfg["long_reasoning"] == "drop":
                stats["reasoning_fix"]["dropped_long"] += 1
                continue
            if cfg["long_reasoning"] == "strip":
                stats["reasoning_fix"]["stripped_long"] += 1
                r["reasoning"] = None
        # MCQ routing
        if cfg["mcq_rl_fraction"] and is_mcq(r) and frac_hash(r["id"]) < cfg["mcq_rl_fraction"]:
            rl_mcq.append(r)
            continue
        kept.append(r)
    stats["level_fit_rate_before_fix"] = round(fit_before[True] / max(1, sum(fit_before.values())), 4)

    # ---------------- open-QA eval ----------------
    rng.shuffle(kept)
    open_eval, train_pool = [], kept
    if not frozen:
        level_n = Counter(r["level"] for r in kept)
        quota = {k: min(cfg["qa_eval_per_level"], int(v * 0.05)) for k, v in level_n.items()}
        per, train_pool = Counter(), []
        for r in kept:
            if per[r["level"]] < quota[r["level"]] and not r.get("messages"):
                per[r["level"]] += 1
                open_eval.append(r)
            else:
                train_pool.append(r)

    # ---------------- balancing ----------------
    by_src = defaultdict(list)
    for r in train_pool:
        by_src[r["source"]].append(r)
    capped = []
    for src, rs in by_src.items():
        cap = cfg["source_caps"].get(src, cfg["source_cap_default"])
        if cfg["reasoning_source_cap"] and sum(1 for r in rs if r.get("reasoning")) > len(rs) / 2:
            cap = min(cap, cfg["reasoning_source_cap"])
        if len(rs) > cap:
            stats["dropped"]["source_cap"] += len(rs) - cap
        capped += rs[:cap]
    by_lvl = defaultdict(list)
    for r in capped:
        by_lvl[r["level"]].append(r)
    final_sft = []
    for lvl, rs in by_lvl.items():
        cap = cfg["level_caps"].get(lvl, cfg["level_cap_default"])
        if len(rs) > cap:
            stats["dropped"]["level_cap"] += len(rs) - cap
            rng.shuffle(rs)
            rs.sort(key=lambda r: 0 if (r["source"] in PRIORITY or r.get("reasoning")) else 1)
        final_sft += rs[:cap]
    ai_sft_n = len(final_sft)
    fit_after = sum(fits(r["level"], r["answer"])[0] for r in final_sft)
    plain = [r for r in final_sft if r["level"] in PLAIN_LEVELS]
    plain_fit = sum(fits(r["level"], r["answer"])[0] for r in plain) / max(1, len(plain))

    # ---------------- replay ----------------
    replay = load_bucket(args.split, "general_replay")
    replay = [r for r in replay if not r["source"].startswith("eval_") and r.get("bucket") != "ai_eval"
              and r.get("answer") and good_sft(r, langs)]
    rng.shuffle(replay)
    n_replay = min(len(replay), int(ai_sft_n * cfg["replay_ratio"]))
    for r in replay[:n_replay]:
        r["level"] = "none" if cfg["replay_system_prompt"] == "none" else cfg["replay_system_prompt"]
        r["topics"] = ["general_replay"]
        r["source"] = f"replay:{r['source']}"
        if r.get("reasoning") and cfg["max_reasoning_chars"] and not reasoning_ok(r, cfg["max_reasoning_chars"]):
            r["reasoning"] = None
        final_sft.append(r)

    # ---------------- general reasoning (non-AI) ----------------
    n_reasoning, reason_cat = 0, Counter()
    if cfg["reasoning_ratio"] > 0:
        by_cat = defaultdict(list)
        per_src = defaultdict(list)
        for r in load_bucket(args.split, "reasoning_sft"):
            per_src[r["source"]].append(r)
        for src_name, rs in sorted(per_src.items()):
            rng.shuffle(rs)
            taken = 0
            for r in rs:
                if taken >= cfg["reasoning_cap_per_source"]:
                    stats["dropped"]["reasoning_source_cap"] += 1
                    continue
                if not good_sft(r, langs):
                    stats["dropped"]["reasoning_quality"] += 1
                    continue
                if cfg["max_reasoning_chars"] and not reasoning_ok(r, cfg["max_reasoning_chars"]):
                    stats["dropped"]["reasoning_too_long"] += 1   # the trace is the point: never strip
                    continue
                k = norm_key(r["question"])
                if k in seen or r["id"] in exclude_ids or k in exclude_keys:
                    stats["dropped"]["reasoning_duplicate"] += 1
                    continue
                if contaminated(r):
                    stats["dropped"]["reasoning_contaminated"] += 1
                    continue
                seen.add(k)
                cat = (r.get("reasoning_check") or {}).get("category", "general")
                by_cat[cat].append(r)
                taken += 1
        target = int(ai_sft_n * cfg["reasoning_ratio"])
        cats = [c for c in cfg["reasoning_categories"] if by_cat.get(c)] + \
               [c for c in by_cat if c not in cfg["reasoning_categories"]]
        for c in cats:
            rng.shuffle(by_cat[c])
        pos = Counter()
        while n_reasoning < target and any(pos[c] < len(by_cat[c]) for c in cats):
            for c in cats:                     # round-robin keeps the categories balanced
                if n_reasoning >= target or pos[c] >= len(by_cat[c]):
                    continue
                r = by_cat[c][pos[c]]
                pos[c] += 1
                r["level"] = "none"
                r["source"] = f"reasoning:{r['source']}"
                final_sft.append(r)
                reason_cat[c] += 1
                n_reasoning += 1
        stats["reasoning_available"] = {c: len(v) for c, v in by_cat.items()}

    rng.shuffle(final_sft)
    n_val = int(len(final_sft) * cfg["val_frac"])
    rows = [to_sft(r, cfg["reasoning"]) for r in final_sft]
    with open(os.path.join(args.out, "sft_val.jsonl"), "w") as fv, \
            open(os.path.join(args.out, "sft_train.jsonl"), "w") as ft:
        for i, row in enumerate(rows):
            (fv if i < n_val else ft).write(json.dumps(row, ensure_ascii=False) + "\n")

    def sys_prompt(level):
        return [{"role": "system", "content": LEVELS[level]["system"]}] if level in LEVELS else []

    # ---------------- RL math ----------------
    rl_math, seen_m = [], set()
    for r in load_bucket(args.split, "ai_rl_math"):
        k = norm_key(r["question"])
        if not r.get("ground_truth") or k in seen_m or contaminated(r):
            stats["dropped"]["rl_math_filtered"] += 1
            continue
        seen_m.add(k)
        rl_math.append(r)
    rng.shuffle(rl_math)
    rl_math = rl_math[:cfg["rl_math_cap"]]
    with open(os.path.join(args.out, "rl_math.jsonl"), "w") as f:
        for r in rl_math:
            f.write(json.dumps({"id": r["id"], "source": r["source"], "level": r["level"],
                                "prompt": sys_prompt(r["level"]) + [{"role": "user", "content": r["question"].strip() + MATH_SUFFIX}],
                                "ground_truth": r["ground_truth"], "meta": r.get("meta", {})}, ensure_ascii=False) + "\n")

    # ---------------- RL code ----------------
    rl_code, seen_c = [], set()
    for r in load_bucket(args.split, "ai_rl_code"):
        k = norm_key(r["question"])
        if not r.get("tests") or k in seen_c or contaminated(r):
            stats["dropped"]["rl_code_filtered"] += 1
            continue
        seen_c.add(k)
        rl_code.append(r)
    rl_code.sort(key=lambda r: 0 if r["ai"]["kw"]["core"] + r["ai"]["kw"]["code"] > 0 else 1)
    rl_code = rl_code[:cfg["rl_code_cap"]]
    with open(os.path.join(args.out, "rl_code.jsonl"), "w") as f:
        for r in rl_code:
            meta = r.get("meta") or {}
            f.write(json.dumps({"id": r["id"], "source": r["source"], "level": r["level"],
                                "prompt": sys_prompt(r["level"]) + [{"role": "user", "content": r["question"].strip() + CODE_SUFFIX}],
                                "tests": r["tests"], "test_format": meta.get("test_format", "pytest"),
                                "test_info": meta.get("test_info"), "reference": r.get("answer")},
                               ensure_ascii=False) + "\n")

    # ---------------- RL MCQ ----------------
    rng.shuffle(rl_mcq)
    rl_mcq = rl_mcq[:cfg["rl_mcq_cap"]]
    with open(os.path.join(args.out, "rl_mcq.jsonl"), "w") as f:
        for r in rl_mcq:
            q = r["question"].strip()
            q = q if "\\boxed" in q else q + MCQ_SUFFIX
            f.write(json.dumps({"id": r["id"], "source": r["source"], "level": r["level"],
                                "prompt": sys_prompt(r["level"]) + [{"role": "user", "content": q}],
                                "ground_truth": str(r["ground_truth"]).strip()}, ensure_ascii=False) + "\n")

    # ---------------- eval ----------------
    if frozen:
        for f in glob.glob(os.path.join(frozen, "*.jsonl")):
            shutil.copy2(f, os.path.join(args.out, "eval", os.path.basename(f)))
        n_open = sum(1 for _ in read_jsonl(os.path.join(args.out, "eval", "open_qa.jsonl")))
        n_evals = sum(sum(1 for _ in read_jsonl(os.path.join(args.out, "eval", f))) for f in ("mcq.jsonl", "code.jsonl"))
    else:
        with open(os.path.join(args.out, "eval", "mcq.jsonl"), "w") as fm, \
                open(os.path.join(args.out, "eval", "code.jsonl"), "w") as fc:
            for r in evals:
                row = {"id": r["id"], "source": r["source"], "question": r["question"], "answer": r.get("answer"),
                       "ground_truth": r.get("ground_truth"), "tests": r.get("tests"), "level": r.get("level"),
                       "meta": r.get("meta", {})}
                (fc if r["task"] == "code" else fm).write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(args.out, "eval", "open_qa.jsonl"), "w") as f:
            for r in open_eval:
                f.write(json.dumps({"id": r["id"], "source": r["source"], "level": r["level"], "topics": r["topics"],
                                    "system": LEVELS[r["level"]]["system"], "question": r["question"],
                                    "reference": r["answer"]}, ensure_ascii=False) + "\n")
        n_open, n_evals = len(open_eval), len(evals)

    # ---------------- stats ----------------
    ai_rows = [r for r in final_sft if not r["source"].startswith(("replay:", "reasoning:"))]
    lv = Counter(r["level"] for r in ai_rows)
    tp = Counter(t for r in final_sft for t in r.get("topics", []))
    src = Counter(r["source"] for r in final_sft)
    n_reason_ai = sum(1 for r in ai_rows if r.get("reasoning") and cfg["reasoning"] == "think")
    stats.update({
        "sft_total": len(rows), "sft_val": n_val, "sft_with_reasoning": sum(r["has_reasoning"] for r in rows),
        "ai_sft": ai_sft_n, "ai_sft_with_reasoning": n_reason_ai,
        "reasoning_share": round(n_reason_ai / max(1, ai_sft_n), 4),
        "level_fit_rate": round(fit_after / max(1, ai_sft_n), 4),
        "plain_level_fit_rate": round(plain_fit, 4), "plain_level_records": len(plain),
        "replay": n_replay, "reasoning_general": n_reasoning, "reasoning_categories": dict(reason_cat),
        "rl_math": len(rl_math), "rl_code": len(rl_code), "rl_mcq": len(rl_mcq),
        "eval_mcq_code": n_evals, "eval_open_qa": n_open, "eval_frozen_from": cfg["frozen_eval"],
        "levels": dict(lv), "topics": dict(tp), "sources": dict(src),
        "reasoning_by_source": dict(Counter(r["source"] for r in ai_rows if r.get("reasoning"))),
        "rl_code_sources": dict(Counter(r["source"] for r in rl_code)),
        "rl_mcq_sources": dict(Counter(r["source"] for r in rl_mcq)),
    })
    for k in ("dropped", "level_fix", "reasoning_fix"):
        stats[k] = dict(stats[k])
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out, "mix_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    L = [f"# Final training mix {cfg['version']}", "",
         f"- SFT: **{len(rows):,}** (val {n_val:,}; AI {ai_sft_n:,}; general replay {n_replay:,}, system prompt: "
         f"{cfg['replay_system_prompt']}; general reasoning {n_reasoning:,} {dict(reason_cat)})",
         f"- AI SFT with reasoning traces: **{n_reason_ai:,}** ({stats['reasoning_share']:.1%})",
         f"- Level fit of AI SFT answers: before fix {stats['level_fit_rate_before_fix']:.1%} -> after **{stats['level_fit_rate']:.1%}**; "
         f"plain-language levels (primary/middle/high/general public): **{stats['plain_level_fit_rate']:.1%}** of {stats['plain_level_records']:,}",
         f"- RL math: **{len(rl_math):,}** | RL code: **{len(rl_code):,}** | RL MCQ: **{len(rl_mcq):,}**",
         f"- Eval: MCQ+code **{n_evals:,}**, open QA **{n_open:,}**" + (f" (frozen from {cfg['frozen_eval']})" if frozen else ""),
         f"- Level fixes: {stats['level_fix']}", f"- Reasoning fixes: {stats['reasoning_fix']}",
         f"- Dropped: {stats['dropped']}", "",
         "| level | SFT |", "|---|---:|"] + [f"| {k} | {lv.get(k, 0):,} |" for k in LEVELS] + \
        ["", "| topic | SFT |", "|---|---:|"] + [f"| {k} | {tp.get(k, 0):,} |" for k in list(TOPICS) + ["general_replay"]] + \
        [f"| reasoning_{k} | {v:,} |" for k, v in reason_cat.most_common()] + \
        ["", "| source | SFT |", "|---|---:|"] + [f"| {k} | {v:,} |" for k, v in src.most_common()]
    with open(os.path.join(args.out, "stats.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
