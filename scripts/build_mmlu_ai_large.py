#!/usr/bin/env python3
"""Build the "ai_large" subject: 1,000 MMLU-style machine-learning / AI questions (8 topics x 125) in the local MMLU
layout, so scripts/eval_model.py evaluates it like any MMLU subject (reported separately as ai_large_accuracy).

    python scripts/build_mmlu_ai_large.py check     # validate the source files, list duplicates / MMLU look-alikes
    python scripts/build_mmlu_ai_large.py blind     # answer-free copies for blind verification -> verify/blind/
    python scripts/build_mmlu_ai_large.py build     # -> data/eval/mmlu/ai_large/{test,dev}-00000-of-00001.parquet

Source (data/eval/mmlu_ai_large/)
    GENERATION_SPEC.md     how the items were written (one writer per topic, 130 test + 1 dev item each)
    source/<topic>.jsonl   the written items: question, 4 choices, answer index, subtopic, difficulty, kind, ...
    verify/blind/          answer-free copies with shuffled choices (verify/perm/ maps them back) for the verifiers
    verify/<topic>.jsonl   blind answers of an independent verifier: {"id", "answer": 0-3 in the blind order or null,
                           "why": "...", "issue": "..."}
    review.json            decisions on the items the verifier disagreed with or flagged:
                           {"<id>": {"action": "keep" | "drop" | "fix", "answer": <index>, "note": "..."}}

Build
    - items whose verifier answer differs from the key, or that carry an issue, need a review.json decision; an
      undecided disputed item is dropped, as are "drop" decisions, exact duplicates and statement-pair items with
      the wrong choices
    - 125 test items per topic, picked at random (seeded) from the rest; a topic short of 125 is topped up from the
      others so the subject has exactly --n items
    - choices are shuffled so the answer letters are balanced (25% each); statement-pair items keep the MMLU
      choice order ("True, True", "False, False", "True, False", "False, True") and the other items make up for them
    - dev split: the writers' dev items (eval_model.py uses the first k as k-shot examples)
    - parquet columns: question, subject ("ai_large"), choices, answer, topic, plus id / subtopic / difficulty /
      kind / explanation for reports; report.md summarizes the build
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "eval", "mmlu_ai_large")
OUT = os.path.join(ROOT, "data", "eval", "mmlu", "ai_large")
MMLU = os.path.join(ROOT, "data", "eval", "mmlu")
TOPICS = ["foundations", "classical_ml", "probabilistic_math", "optimization_training", "architectures", "llm_nlp",
          "rl_classical_ai", "applied_responsible"]
PAIR_CHOICES = ["True, True", "False, False", "True, False", "False, True"]
REQUIRED = ("id", "split", "question", "choices", "answer")
LOOKALIKE_SUBJECTS = ["machine_learning", "college_computer_science", "high_school_computer_science",
                      "computer_security", "high_school_statistics", "college_mathematics"]


def read_jsonl(path: str) -> tuple[list[dict], list[str]]:
    rows, errors = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    errors.append(f"{os.path.basename(path)}:{i}: bad JSON ({e})")
    return rows, errors


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def words(text: str) -> set[str]:
    return {w for w in norm(text).split() if len(w) > 2}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def load_source() -> tuple[list[dict], list[str]]:
    """All items of all topics, with "topic" set, plus a list of problems (bad JSON / fields / choices)."""
    items, problems = [], []
    for topic in TOPICS:
        path = os.path.join(SRC, "source", f"{topic}.jsonl")
        if not os.path.exists(path):
            problems.append(f"{topic}: missing {os.path.relpath(path, ROOT)}")
            continue
        rows, errors = read_jsonl(path)
        problems += errors
        for r in rows:
            r["topic"] = topic
            miss = [k for k in REQUIRED if k not in r]
            if miss:
                problems.append(f"{r.get('id', topic)}: missing {miss}")
                continue
            ch = r["choices"]
            if not (isinstance(ch, list) and len(ch) == 4 and all(isinstance(c, str) and c.strip() for c in ch)):
                problems.append(f"{r['id']}: needs 4 non-empty choices")
                r["invalid"] = True
            elif len({" ".join(c.lower().split()) for c in ch}) < 4:  # exact text: "-x" and "x" differ
                problems.append(f"{r['id']}: repeated choice")
                r["invalid"] = True
            if not (isinstance(r["answer"], int) and 0 <= r["answer"] < 4):
                problems.append(f"{r['id']}: answer must be 0-3")
                r["invalid"] = True
            if r.get("kind") == "statement_pair" or r.get("fixed_order"):
                if ch != PAIR_CHOICES:
                    problems.append(f"{r['id']}: statement_pair choices must be {PAIR_CHOICES}")
                    r["invalid"] = True
                r["fixed_order"] = True
            if re.search(r"\b(all|none|both) of (the )?(above|these)\b|\b(both|either) [abcd] and [abcd]\b",
                         " ".join(ch), re.IGNORECASE):
                problems.append(f"{r['id']}: positional option (all / none of the above, both A and B)")
            items.append(r)
    ids = Counter(r["id"] for r in items)
    problems += [f"{i}: id used {n} times" for i, n in ids.items() if n > 1]
    return items, problems


def duplicates(items: list[dict]) -> tuple[set[str], list[tuple]]:
    """(ids of exact duplicates to drop: the later copy, near-duplicate pairs (Jaccard >= 0.6) to look at)."""
    seen, exact = {}, set()
    for r in items:
        k = norm(r["question"]) + "|" + "|".join(sorted(norm(c) for c in r["choices"]))
        if k in seen:
            exact.add(r["id"])
        else:
            seen[k] = r["id"]
    ws = [(r["id"], words(r["question"] + " " + " ".join(r["choices"]))) for r in items]
    near = []
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            s = jaccard(ws[i][1], ws[j][1])
            if s >= 0.6 and ws[j][0] not in exact:
                near.append((ws[i][0], ws[j][0], round(s, 2)))
    return exact, near


def mmlu_lookalikes(items: list[dict], threshold: float = 0.5) -> list[tuple]:
    """Items that resemble a question of the related MMLU subjects (Jaccard of question+choice words)."""
    import pyarrow.parquet as pq

    ref = []
    for s in LOOKALIKE_SUBJECTS:
        for f in glob.glob(os.path.join(MMLU, s, "*.parquet")):
            for r in pq.read_table(f, columns=["question", "choices"]).to_pylist():
                ref.append((s, r["question"][:80], words(r["question"] + " " + " ".join(r["choices"]))))
    hits = []
    for r in items:
        w = words(r["question"] + " " + " ".join(r["choices"]))
        best = max(((jaccard(w, rw), s, q) for s, q, rw in ref), default=(0, "", ""))
        if best[0] >= threshold:
            hits.append((r["id"], best[1], round(best[0], 2), best[2]))
    return hits


def load_verification() -> dict[str, dict]:
    """Verifier answers by id, with "answer" mapped from the shuffled blind order back to the source order."""
    out = {}
    for f in sorted(glob.glob(os.path.join(SRC, "verify", "*.jsonl"))):
        topic = os.path.basename(f)[:-len(".jsonl")]
        with open(os.path.join(SRC, "verify", "perm", f"{topic}.json"), encoding="utf-8") as pf:
            perm = json.load(pf)  # id -> source index of each blind position
        rows, _ = read_jsonl(f)
        for r in rows:
            if "id" in r:
                a = r.get("answer")
                out[r["id"]] = {**r, "answer": perm[r["id"]][a] if isinstance(a, int) and 0 <= a < 4 else None}
    return out


def disputed(items: list[dict], ver: dict) -> list[dict]:
    """Test items the verifier answered differently, could not answer, or flagged with an issue."""
    out = []
    for r in items:
        v = ver.get(r["id"])
        if r["split"] != "test" or v is None:
            continue
        if v.get("answer") != r["answer"] or (v.get("issue") or "").strip():
            out.append({"id": r["id"], "key": r["answer"], "verifier": v.get("answer"), "issue": v.get("issue")})
    return out


def cmd_check(args):
    items, problems = load_source()
    exact, near = duplicates(items)
    look = mmlu_lookalikes(items)
    ver = load_verification()
    print(f"{len(items)} items in {len({r['topic'] for r in items})} topics; "
          f"{sum(r['split'] == 'test' for r in items)} test, {sum(r['split'] == 'dev' for r in items)} dev")
    for t in TOPICS:
        rs = [r for r in items if r["topic"] == t]
        print(f"  {t:22s} {len(rs):4d}  difficulty {dict(Counter(r.get('difficulty') for r in rs))}  "
              f"kind {dict(Counter(r.get('kind') for r in rs))}")
    free = [r for r in items if not r.get("fixed_order") and not r.get("invalid")]
    longest = sum(len(r["choices"][r["answer"]]) > max(len(c) for i, c in enumerate(r["choices"]) if i != r["answer"])
                  for r in free)
    print(f"surface cue: the key is strictly the longest option in {longest}/{len(free)} free items "
          f"({longest / max(1, len(free)):.1%}; chance ~25%)")
    for t in TOPICS:
        fr = [r for r in free if r["topic"] == t]
        n = sum(len(r["choices"][r["answer"]]) > max(len(c) for i, c in enumerate(r["choices"]) if i != r["answer"])
                for r in fr)
        print(f"  {t:22s} {n}/{len(fr)}")
    print(f"\nproblems ({len(problems)}):", *problems[:200], sep="\n  ")
    print(f"\nexact duplicates dropped ({len(exact)}):", *sorted(exact), sep="\n  ")
    print(f"\nnear duplicates ({len(near)}):", *near, sep="\n  ")
    print(f"\nMMLU look-alikes ({len(look)}):", *look, sep="\n  ")
    if ver:
        d = disputed(items, ver)
        n = sum(1 for r in items if r["split"] == "test" and r["id"] in ver)
        agree = n - sum(1 for x in d if x["verifier"] != x["key"])
        print(f"\nverification: {n} test items answered blind, agreement with the key {agree}/{n} "
              f"({agree / max(1, n):.1%}); {len(d)} disputed or flagged")


def cmd_blind(args):
    """Answer-free copies with the choices shuffled (the writers put the key first), plus the permutation that maps
    each blind position back to the source choice (verify/perm/, read by load_verification)."""
    items, _ = load_source()
    for d in ("blind", "perm"):
        os.makedirs(os.path.join(SRC, "verify", d), exist_ok=True)
    for t in (args.topics.split(",") if args.topics else TOPICS):
        rng = random.Random(f"{args.seed}-{t}")
        rows, perm = [], {}
        for r in items:
            if r["topic"] != t or r["split"] != "test":
                continue
            order = list(range(len(r["choices"])))
            if not r.get("fixed_order"):
                rng.shuffle(order)
            perm[r["id"]] = order
            rows.append({"id": r["id"], "question": r["question"], "choices": [r["choices"][i] for i in order]})
        with open(os.path.join(SRC, "verify", "blind", f"{t}.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        with open(os.path.join(SRC, "verify", "perm", f"{t}.json"), "w", encoding="utf-8") as f:
            json.dump(perm, f)
        print(f"{t}: {len(rows)} blind items")


def balance(rows: list[dict], rng: random.Random) -> None:
    """Shuffle each free item's choices so the answers of all rows spread evenly over A-D (in place)."""
    fixed = Counter(r["answer"] for r in rows if r.get("fixed_order"))
    free = [r for r in rows if not r.get("fixed_order")]
    n = len(rows)
    want = [n // 4 + (1 if k < n % 4 else 0) - fixed[k] for k in range(4)]
    while min(want) < 0:  # more fixed answers at one position than its share: move the excess to the others
        k = want.index(min(want))
        want[k] += 1
        want[want.index(max(want))] -= 1
    targets = [k for k in range(4) for _ in range(want[k])]
    rng.shuffle(targets)
    for r, t in zip(free, targets):
        right = r["choices"][r["answer"]]
        wrong = [c for i, c in enumerate(r["choices"]) if i != r["answer"]]
        rng.shuffle(wrong)
        r["choices"] = wrong[:t] + [right] + wrong[t:]
        r["answer"] = t


def cmd_build(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = random.Random(args.seed)
    items, problems = load_source()
    exact, near = duplicates(items)
    ver = load_verification()
    review = {}
    if os.path.exists(os.path.join(SRC, "review.json")):
        with open(os.path.join(SRC, "review.json"), encoding="utf-8") as f:
            review = json.load(f)
    disp = {d["id"]: d for d in disputed(items, ver)}
    n_ver = sum(1 for r in items if r["split"] == "test" and r["id"] in ver)
    agree = sum(1 for r in items if r["split"] == "test" and r["id"] in ver and ver[r["id"]].get("answer") == r["answer"])
    dropped = Counter()
    pool = defaultdict(list)
    dev = []
    for r in items:
        dec = review.get(r["id"], {})
        reason = ("invalid" if r.get("invalid") else "duplicate" if r["id"] in exact
                  else "review: drop" if dec.get("action") == "drop"
                  else "disputed, undecided" if r["id"] in disp and dec.get("action") not in ("keep", "fix") else None)
        if reason:
            dropped[reason] += 1
            continue
        if dec.get("action") == "fix":
            r = {**r, **{k: dec[k] for k in ("answer", "question", "choices") if k in dec}}
        (dev if r["split"] == "dev" else pool[r["topic"]]).append(r)
    if ver and not args.allow_unverified:
        unv = [r["id"] for rs in pool.values() for r in rs if r["id"] not in ver]
        if unv:
            sys.exit(f"{len(unv)} test items have no verifier answer (e.g. {unv[:5]}); --allow-unverified to build anyway")

    per = args.n // len(TOPICS)
    test, spare = [], []
    for t in TOPICS:
        rs = sorted(pool[t], key=lambda r: r["id"])
        rng.shuffle(rs)
        test += rs[:per]
        spare += rs[per:]
    rng.shuffle(spare)
    test += spare[:args.n - len(test)]
    if len(test) < args.n:
        print(f"! only {len(test)} usable test items (wanted {args.n})")
    test.sort(key=lambda r: (TOPICS.index(r["topic"]), r["id"]))
    dev.sort(key=lambda r: TOPICS.index(r["topic"]))
    balance(test, rng)
    balance(dev, rng)

    os.makedirs(OUT, exist_ok=True)
    cols = ["question", "subject", "choices", "answer", "topic", "id", "subtopic", "difficulty", "kind", "explanation"]
    for split, rows in (("test", test), ("dev", dev)):
        recs = [{"question": r["question"].strip(), "subject": "ai_large", "choices": [c.strip() for c in r["choices"]],
                 "answer": int(r["answer"]), "topic": r["topic"], "id": r["id"], "subtopic": r.get("subtopic") or "",
                 "difficulty": r.get("difficulty") or "", "kind": r.get("kind") or "",
                 "explanation": r.get("explanation") or ""} for r in rows]
        pq.write_table(pa.Table.from_pylist(recs).select(cols), os.path.join(OUT, f"{split}-00000-of-00001.parquet"))

    L = ["# ai_large: build report", "",
         f"{len(test)} test and {len(dev)} dev items in {OUT.replace(ROOT + '/', '')}/ (seed {args.seed}).", "",
         f"- source items: {len(items)} ({sum(r['split'] == 'test' for r in items)} test)",
         f"- blind verification: {n_ver} test items, verifier agreed with the key on {agree} "
         f"({agree / max(1, n_ver):.1%}); {len(disp)} disputed or flagged, decisions in review.json "
         f"({Counter(v.get('action') for v in review.values())})",
         f"- not used: {dict(dropped)}; near-duplicate pairs looked at: {len(near)}",
         f"- answer letters: {dict(sorted(Counter('ABCD'[r['answer']] for r in test).items()))}", "",
         "| topic | n | easy | medium | hard | concept | calculation | scenario | statement_pair |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in TOPICS:
        rs = [r for r in test if r["topic"] == t]
        d, k = Counter(r.get("difficulty") for r in rs), Counter(r.get("kind") for r in rs)
        L.append(f"| {t} | {len(rs)} | {d['easy']} | {d['medium']} | {d['hard']} | {k['concept']} | {k['calculation']} "
                 f"| {k['scenario']} | {k['statement_pair']} |")
    with open(os.path.join(SRC, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "blind", "build"])
    ap.add_argument("--n", type=int, default=1000, help="test items in the subject")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topics", default=None, help="blind: only these topics (comma list); the others are left as they are")
    ap.add_argument("--allow-unverified", action="store_true", help="build although some items were not verified")
    args = ap.parse_args()
    {"check": cmd_check, "blind": cmd_blind, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
