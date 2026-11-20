#!/usr/bin/env python3
"""Step 2 - split: decide which collected records are AI-related, and label them.

Cascade (cheapest first, every record stops at the first stage that is confident):

  stage 1  metadata rule   hints.meta_decision from the source's own fields
                           (stackexchange site, MMLU subject, DeepMath topic, cosmopedia audience, ...)
  stage 2  keywords        weighted lexicon -> accept / reject / uncertain
  stage 3  zero-shot NLI   deberta-v3-base-zeroshot-v2.0 on the uncertain records
                           (and on keyword-accepts too with --verify-accepts)

Records that fail the AI test but come from a source with reasoning=True (or domain "reasoning") and pass
ai_qa_data/reasoning.py (complete trace, fits the context, answer matches the ground truth) go to
reasoning_sft/ instead of being dropped: general math / code / logic / science reasoning, no level label.

Accepted records are then labelled:
  * topics  (concepts / applications / theory_math / history / coding / ethics_society / research)
  * level   (primary_school ... ai_researcher), see ai_qa_data/levels.py

    python scripts/split_ai_data.py                         # all files in data/raw
    python scripts/split_ai_data.py --sources se_ai,eli5     # subset
    python scripts/split_ai_data.py --label-mode fast        # no zero-shot for topic/level labels
    python scripts/split_ai_data.py --no-zeroshot            # keyword only (uncertain -> rejected)

Output (--out, default data/split):
    ai_sft/<source>.jsonl  ai_rl_math/<source>.jsonl  ai_rl_code/<source>.jsonl  ai_eval/<source>.jsonl
    reasoning_sft/<source>.jsonl      non-AI reasoning examples (math / code / logic / science)
    general_replay/<source>.jsonl     non-AI sample for anti-forgetting
    audit/<source>.jsonl              borderline zero-shot decisions for manual inspection
    report.json / report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_qa_data.keywords import keyword_decision, score_record  # noqa: E402
from ai_qa_data.levels import infer_level  # noqa: E402
from ai_qa_data.reasoning import DEFAULT_MAX_CHARS, reasoning_check, to_reasoning_record  # noqa: E402
from ai_qa_data.sources import SOURCES_BY_NAME  # noqa: E402
from ai_qa_data.taxonomy import DOMAIN_HYPOTHESES, LEVELS, TOPICS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_RE = re.compile(r"```|^\s*(?:def |class |import |from \w+ import |for .+ in .+:|print\()|\w+\.\w+\(.*\)\s*$",
                     re.MULTILINE)
BUCKETS = ("ai_sft", "ai_rl_math", "ai_rl_code", "ai_eval", "reasoning_sft", "general_replay", "audit")


def premise(rec: dict, n_q: int = 1200, n_a: int = 600) -> str:
    q = rec["question"][:n_q]
    a = (rec.get("answer") or rec.get("reasoning") or "")[:n_a]
    return f"{q}\n{a}" if a else q


def _kw_job(rec):
    kw = score_record(rec["question"], rec.get("answer") or (rec.get("reasoning") or "")[:3000])
    return kw.to_dict(), keyword_decision(kw, rec.get("domain", "ai_core"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def chunks(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


class Splitter:
    def __init__(self, args):
        self.args = args
        self.zs = None
        if not args.no_zeroshot:
            from ai_qa_data.zeroshot import ZeroShotClassifier
            self.zs = ZeroShotClassifier(args.model, batch_size=args.zs_batch, max_length=args.zs_max_len)
        self.thr = {"ai_core": args.zs_core, "ai_math": args.zs_math, "ai_code": args.zs_code}
        self.pool = Pool(args.kw_workers) if args.kw_workers > 1 else None

    # ---------------- relevance -----------------
    def relevance_zs(self, recs: list[dict]) -> list[dict]:
        out = [None] * len(recs)
        by_dom = defaultdict(list)
        for i, r in enumerate(recs):
            by_dom[r.get("domain", "ai_core")].append(i)
        for dom, idx in by_dom.items():
            labels = ["ai_core"] + ([dom] if dom != "ai_core" else [])
            hyps = {k: DOMAIN_HYPOTHESES[k] for k in labels}
            probs = self.zs.multi_label([premise(recs[i]) for i in idx], hyps)
            for i, p in zip(idx, probs):
                out[i] = p
        return out

    def is_relevant(self, probs: dict) -> bool:
        return any(v >= self.thr[k] for k, v in probs.items())

    # ---------------- labels -----------------
    def label(self, recs: list[dict]):
        """Topics: zero-shot (>= --topic-thr) OR strong keyword evidence; `coding` is deterministic
        (NLI models over-entail "contains code" for any technical text).
        Level: metadata prior + readability + cues, plus a zero-shot vote unless the prior is confident."""
        n = len(recs)
        zs_topics, zs_levels = [None] * n, [None] * n
        if self.zs is not None and self.args.label_mode == "full":
            texts = [premise(r, 800, 800) for r in recs]
            hyps = {k: v for k, v in TOPICS.items() if k != "coding"}
            zs_topics = self.zs.multi_label(texts, hyps, max_length=self.args.label_max_len)
            need = [i for i, r in enumerate(recs)
                    if max((r["hints"].get("level_prior") or {"_": 0}).values()) < 0.7]
            if need:
                lv = self.zs.single_label([texts[i] for i in need], {k: v["zs"] for k, v in LEVELS.items()},
                                          max_length=self.args.label_max_len)
                for i, p in zip(need, lv):
                    zs_levels[i] = p
        for r, tp, lp in zip(recs, zs_topics, zs_levels):
            hints = r["ai"]["kw"]["topic_hints"]
            scores = {}
            for t in TOPICS:
                s_kw = min(1.0, hints.get(t, 0.0) / 8.0)
                s_zs = (tp or {}).get(t, 0.0)
                # zero-shot is calibrated differently from keyword evidence: rescale to a common 0.5 cut
                s_zs_cal = 0.5 + (s_zs - self.args.topic_thr) / (2 * (1 - self.args.topic_thr)) if s_zs >= self.args.topic_thr \
                    else 0.5 * s_zs / self.args.topic_thr
                scores[t] = round(max(s_kw, s_zs_cal), 3)
            has_code = bool(CODE_RE.search(r["question"]) or CODE_RE.search(r.get("answer") or ""))
            scores["coding"] = 1.0 if (has_code or r["task"] in ("code", "rl_code")) else 0.0
            topics = {t for t, v in scores.items() if v >= 0.5}
            if r["task"] == "rl_math" or r.get("domain") == "ai_math":
                topics.add("theory_math")
            if not topics:
                topics.add("concepts")
            r["topics"] = sorted(topics)
            r["topic_scores"] = scores
            level, lscores = infer_level(r["question"], r.get("answer"), r["hints"].get("level_prior"), lp)
            r["level"] = level
            r["level_scores"] = lscores
            if lp:
                r["level_zs"] = lp

    # ---------------- one source -----------------
    def run_source(self, path: str, out_dir: str) -> dict:
        name = os.path.basename(path)[:-len(".jsonl")]
        done = os.path.join(out_dir, "_done", f"{name}.json")
        if os.path.exists(done) and not self.args.force:
            with open(done) as f:
                return json.load(f)
        rng = random.Random(0)
        st = Counter()
        writers = {}

        def w(bucket, rec):
            if bucket not in writers:
                writers[bucket] = open(os.path.join(out_dir, bucket, f"{name}.jsonl.tmp"), "w", encoding="utf-8")
            writers[bucket].write(json.dumps(rec, ensure_ascii=False) + "\n")
            st[f"out_{bucket}"] += 1

        level_c, topic_c, reason_c = Counter(), Counter(), Counter()
        src = SOURCES_BY_NAME.get(name)
        reasoning_on = bool(src and src.reasoning) and not self.args.no_reasoning

        def to_reasoning(rec) -> bool:
            """Non-AI record: keep it as a general reasoning example if it passes the reasoning check."""
            if not reasoning_on:
                return False
            chk = rec["hints"].get("reasoning")
            if chk is None or chk.get("chars", 0) > self.args.reasoning_max_chars:
                chk = reasoning_check(rec, self.args.reasoning_max_chars)
            if not chk["ok"]:
                st[f"reasoning_fail_{chk['why']}"] += 1
                return False
            reason_c[chk["category"]] += 1
            w("reasoning_sft", to_reasoning_record(rec, chk))
            return True

        replay_budget = self.args.replay_per_source
        if name.startswith("eval_"):  # never let benchmark items leak into (replay) training data
            replay_budget = 0
        t0 = time.time()
        for batch in chunks(read_jsonl(path), self.args.chunk):
            st["in"] += len(batch)
            # stage 1 + 2
            kws = self.pool.map(_kw_job, batch, chunksize=64) if self.pool else list(map(_kw_job, batch))
            accepted, uncertain = [], []
            for rec, (kw, dec) in zip(batch, kws):
                rec["ai"] = {"kw": kw}
                md = rec["hints"].get("meta_decision")
                if md == "accept":
                    rec["ai"]["by"] = "meta"
                    st["meta_accept"] += 1
                    accepted.append(rec)
                elif md == "reject":             # kept at collection only for its reasoning value
                    st["meta_reject"] += 1
                    to_reasoning(rec)
                elif dec == "accept" and not self.args.verify_accepts:
                    rec["ai"]["by"] = "keyword"
                    st["kw_accept"] += 1
                    accepted.append(rec)
                elif dec == "reject" or rec.get("domain") == "reasoning":   # reasoning pools: no zero-shot
                    st["kw_reject"] += 1
                    if to_reasoning(rec):
                        continue
                    if replay_budget > 0 and rng.random() < 0.05:
                        replay_budget -= 1
                        w("general_replay", rec)
                else:
                    uncertain.append(rec)
            # stage 3
            if uncertain:
                if self.zs is None:
                    st["kw_uncertain_dropped"] += len(uncertain)
                else:
                    probs = self.relevance_zs(uncertain)
                    for rec, p in zip(uncertain, probs):
                        rec["ai"]["zs"] = p
                        ok = self.is_relevant(p)
                        borderline = any(abs(v - self.thr[k]) < 0.1 for k, v in p.items())
                        if borderline and st["audit"] < self.args.audit_per_source:
                            st["audit"] += 1
                            w("audit", {"id": rec["id"], "accepted": ok, "zs": p, "kw": rec["ai"]["kw"],
                                        "question": rec["question"][:800], "answer": (rec.get("answer") or "")[:400]})
                        if ok:
                            rec["ai"]["by"] = "zeroshot"
                            st["zs_accept"] += 1
                            accepted.append(rec)
                        else:
                            st["zs_reject"] += 1
                            if to_reasoning(rec):
                                continue
                            if replay_budget > 0 and rng.random() < 0.2:
                                replay_budget -= 1
                                w("general_replay", rec)
            # labels + routing
            if accepted:
                self.label(accepted)
                for rec in accepted:
                    b = rec["bucket"]
                    if b == "ai_rl_math" and not rec.get("ground_truth"):
                        b = "ai_sft" if rec.get("answer") else None
                    if b == "ai_rl_code" and not rec.get("tests"):
                        b = "ai_sft" if rec.get("answer") else None
                    if b is None:
                        st["dropped_unusable"] += 1
                        continue
                    rec["bucket"] = b
                    level_c[rec["level"]] += 1
                    topic_c.update(rec["topics"])
                    w(b, rec)
            print(f"  [{name}] {st['in']:,} in | meta+ {st['meta_accept']:,} kw+ {st['kw_accept']:,} "
                  f"zs+ {st['zs_accept']:,} | kw- {st['kw_reject']:,} zs- {st['zs_reject']:,} "
                  f"| reasoning {st['out_reasoning_sft']:,} "
                  f"({time.time() - t0:.0f}s)", flush=True)

        # raw replay sample collected during streaming
        raw_replay = path[:-len(".jsonl")] + ".replay.jsonl"
        if os.path.exists(raw_replay):
            for rec in read_jsonl(raw_replay):
                if replay_budget <= 0:
                    break
                replay_budget -= 1
                w("general_replay", rec)

        for b, fh in writers.items():
            fh.close()
            os.replace(os.path.join(out_dir, b, f"{name}.jsonl.tmp"), os.path.join(out_dir, b, f"{name}.jsonl"))
        res = {"source": name, **st, "levels": dict(level_c), "topics": dict(topic_c),
               "reasoning_categories": dict(reason_c),
               "seconds": round(time.time() - t0, 1)}
        with open(done, "w") as f:
            json.dump(res, f, indent=2)
        return res


def write_report(results: list[dict], out_dir: str):
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(results, f, indent=2)
    L = ["# AI split report", "",
         "| source | in | meta+ | kw+ | zs+ | kw- | zs- | ai_sft | rl_math | rl_code | eval | reasoning | replay |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    tot = Counter()
    for r in sorted(results, key=lambda x: x["source"]):
        row = [r.get(k, 0) for k in ("in", "meta_accept", "kw_accept", "zs_accept", "kw_reject", "zs_reject",
                                     "out_ai_sft", "out_ai_rl_math", "out_ai_rl_code", "out_ai_eval",
                                     "out_reasoning_sft", "out_general_replay")]
        tot.update(dict(zip(range(len(row)), row)))
        L.append(f"| {r['source']} | " + " | ".join(f"{x:,}" for x in row) + " |")
    L.append("| **total** | " + " | ".join(f"**{tot[i]:,}**" for i in range(12)) + " |")
    lv, tp, rc = Counter(), Counter(), Counter()
    for r in results:
        lv.update(r.get("levels", {}))
        tp.update(r.get("topics", {}))
        rc.update(r.get("reasoning_categories", {}))
    L += ["", "## Levels (accepted records)", "", "| level | n |", "|---|---:|"]
    L += [f"| {k} | {lv.get(k, 0):,} |" for k in LEVELS]
    L += ["", "## Topics (multi-label)", "", "| topic | n |", "|---|---:|"]
    L += [f"| {k} | {tp.get(k, 0):,} |" for k in TOPICS]
    L += ["", "## Reasoning SFT (non-AI) by category", "", "| category | n |", "|---|---:|"]
    L += [f"| {k} | {v:,} |" for k, v in rc.most_common()]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "split"))
    ap.add_argument("--sources", default="all")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "deberta-v3-base-zeroshot-v2.0"))
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--verify-accepts", action="store_true",
                    help="also send keyword-accepted records through zero-shot (higher precision, slower)")
    ap.add_argument("--label-mode", choices=["full", "fast"], default="full",
                    help="full = zero-shot topic+level labels; fast = keyword/heuristic labels only")
    ap.add_argument("--zs-core", type=float, default=0.85, help="P(entail) threshold for ai_core")
    ap.add_argument("--zs-math", type=float, default=0.85, help="P(entail) threshold for ai_math")
    ap.add_argument("--zs-code", type=float, default=0.80, help="P(entail) threshold for ai_code")
    ap.add_argument("--topic-thr", type=float, default=0.85, help="zero-shot P(entail) needed for a topic")
    ap.add_argument("--label-max-len", type=int, default=192)
    ap.add_argument("--zs-batch", type=int, default=128)
    ap.add_argument("--zs-max-len", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--kw-workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--replay-per-source", type=int, default=2000)
    ap.add_argument("--audit-per-source", type=int, default=300)
    ap.add_argument("--no-reasoning", action="store_true", help="do not build the reasoning_sft bucket")
    ap.add_argument("--reasoning-max-chars", type=int, default=DEFAULT_MAX_CHARS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    files = sorted(p for p in glob.glob(os.path.join(args.raw, "*.jsonl")) if not p.endswith(".replay.jsonl"))
    if args.sources != "all":
        want = set(args.sources.split(","))
        files = [p for p in files if os.path.basename(p)[:-6] in want]
    if not files:
        sys.exit(f"no raw files in {args.raw}; run collect_datasets.py first")
    for b in BUCKETS + ("_done",):
        os.makedirs(os.path.join(args.out, b), exist_ok=True)

    sp = Splitter(args)
    results = []
    for p in files:
        print(f"== {os.path.basename(p)}", flush=True)
        results.append(sp.run_source(p, args.out))
    # include previously finished sources in the report
    names = {r["source"] for r in results}
    for d in glob.glob(os.path.join(args.out, "_done", "*.json")):
        with open(d) as f:
            r = json.load(f)
        if r["source"] not in names:
            results.append(r)
    write_report(results, args.out)


if __name__ == "__main__":
    main()
