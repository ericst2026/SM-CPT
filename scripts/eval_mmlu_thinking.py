#!/usr/bin/env python3
"""0-shot MMLU in Qwen3 *thinking* mode, for chat models with or without a (CPT) adapter applied to them.

The model reasons inside <think>...</think> and then answers; the letter is parsed from the answer text. The test
for a CPT/SFT conflict: the chat model alone vs the chat model + CPT adapter, on the same questions, the way the
tutor is used.

    python scripts/eval_mmlu_thinking.py \\
        --run chat06b_think=models/Qwen3-0.6B \\
        --run cpt_qwen06b_r64_300mb_epoch1_onchat_think=runs/cpt_qwen06b_r64_300mb/epoch1/adapter@models/Qwen3-0.6B

    --run TAG=MODEL[@BASE]   MODEL = model dir or adapter dir; @BASE puts the adapter on BASE instead of the base in
                             its adapter_config.json (the chat model for CPT adapters). Repeat for several runs.

How it works
  * all test questions are put in one seeded random order and processed in chunks; every run does chunk k before
    any run starts chunk k+1, so after any chunk the runs have answered the same random sample (paired, unbiased)
  * resumable: eval_results/<TAG>/mmlu_thinking_predictions.jsonl is appended per chunk and skipped on restart
  * generation: continuous batching (transformers generate_batch), sampling with Qwen3's thinking-mode settings
    (temperature 0.6, top-p 0.95, top-k 20; Qwen advises against greedy decoding in thinking mode), fixed seed
  * answer: the letter after </think> (\\boxed{X}, "Answer: X", a lone letter, ...). If the reasoning hits
    --max-new-tokens, or no letter can be read, the reasoning is closed with "</think>" and the letter with the
    highest next-token logit is taken ("forced"); the share of forced answers is reported
  * summary.json of each tag gets tasks.mmlu_thinking (other tasks in an existing summary.json are kept)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from ai_qa_train.common import load_model, load_tokenizer, resolve_adapter  # noqa: E402
from eval_model import (LETTERS, MMLU_AI_RELATED, MMLU_AI_SUBJECTS, MMLU_DIR, RESULTS_ROOT, ROOT,  # noqa: E402
                        list_mmlu_subjects, load_ai_related, load_mmlu_split, mmlu_category, mmlu_prompt)

LLM_SELECTION = os.path.join(ROOT, "data", "eval", "mmlu_ai_related_llm", "selection.json")
PRED_FILE = "mmlu_thinking_predictions.jsonl"
THINK_END = "</think>"
_PATTERNS = [r"\\boxed\{\s*\(?(?:\\text\{)?([A-J])\}?\)?\s*\}",          # \boxed{C}, \boxed{\text{C}}
             r"(?i:answer)(?:\s+(?i:is))?\s*[:\-]?\s*\(?([A-J])\)?(?![A-Za-z])",  # Answer: C / the answer is (C)
             r"^\s*\(?([A-J])(?:[.)]|\s*$)",                               # a line "C." / "C)" / "C"
             r"(?i:option|choice)\s+\(?([A-J])\)?(?![A-Za-z])"]          # option C


def parse_letter(answer: str, n_opts: int) -> str | None:
    """First option letter stated in the answer text (case-sensitive, so the article "a" never counts)."""
    text = answer.replace("*", "").replace("$", "").strip()
    valid = set(LETTERS[:n_opts])
    for pat in _PATTERNS:
        for m in re.finditer(pat, text, re.M):
            if m.group(1) in valid:
                return m.group(1)
    return None


def load_questions(split: str, seed: int, limit_per_subject: int | None) -> list[dict]:
    qs = []
    for subj in list_mmlu_subjects(MMLU_DIR):
        rows = load_mmlu_split(MMLU_DIR, subj, split)
        for i, r in enumerate(rows[:limit_per_subject] if limit_per_subject else rows):
            qs.append({"id": f"mmlu/{subj}/{split}/{i}", "subject": subj, "row": r,
                       "gold": LETTERS[int(r["answer"])], "n_opts": len(r["choices"])})
    random.Random(seed).shuffle(qs)
    return qs


class Runner:
    def __init__(self, tag: str, spec: str, args):
        model, _, base_override = spec.partition("@")
        base, adapter = resolve_adapter(os.path.abspath(model))
        if base_override:
            if not adapter:
                raise SystemExit(f"{tag}: @BASE needs an adapter as MODEL")
            base = os.path.abspath(base_override)
        self.tag, self.base, self.adapter = tag, base, adapter
        self.out = os.path.join(RESULTS_ROOT, tag)
        os.makedirs(self.out, exist_ok=True)
        self.tok = load_tokenizer(base)
        self.model = load_model(base, adapter=adapter, merge=True, attn="paged|sdpa")
        self.letter_ids = [self.tok.encode(x, add_special_tokens=False)[0] for x in LETTERS]
        self.close_ids = self.tok.encode("\n" + THINK_END + "\n\n", add_special_tokens=False)
        self.after_close = self.tok.encode("\n\n", add_special_tokens=False)
        self.think_end_id = self.tok.convert_tokens_to_ids(THINK_END)
        self.done = {}
        path = os.path.join(self.out, PRED_FILE)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        p = json.loads(line)
                        self.done[p["id"]] = p
        print(f"[{tag}] base={base} adapter={adapter}  ({len(self.done)} questions already done)", flush=True)

    @torch.inference_mode()
    def forced_letters(self, seqs: list[list[int]], n_opts: list[int]) -> list[str]:
        """Letter with the highest logit after each token sequence (one sequence at a time: they are few)."""
        out = []
        for ids, n in zip(seqs, n_opts):
            x = torch.tensor([ids[-6000:]], device=self.model.device)
            logits = self.model(input_ids=x).logits[0, -1].float()
            out.append(LETTERS[int(torch.argmax(logits[self.letter_ids[:n]]))])
        return out

    def run_chunk(self, chunk: list[dict], args) -> int:
        from transformers import GenerationConfig

        todo = [q for q in chunk if q["id"] not in self.done]
        if not todo:
            return 0
        prompts = []
        for q in todo:
            text = self.tok.apply_chat_template(mmlu_prompt(q["row"], q["subject"], [], "chat"),
                                                add_generation_prompt=True, enable_thinking=True, tokenize=False)
            prompts.append(self.tok(text, add_special_tokens=False)["input_ids"])
        gc = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature,
                              top_p=args.top_p, top_k=args.top_k, eos_token_id=[151645, 151643],
                              pad_token_id=self.tok.pad_token_id)
        torch.manual_seed(args.seed)
        t0 = time.time()
        from transformers.generation.configuration_utils import ContinuousBatchingConfig

        cb = ContinuousBatchingConfig(num_blocks=args.cb_blocks, max_batch_tokens=args.cb_batch_tokens)
        res = self.model.generate_batch(inputs=prompts, generation_config=gc, continuous_batching_config=cb)
        gens = []
        for i, p in enumerate(prompts):
            o = res[f"req_{i}"]
            if list(o.prompt_ids) != p:
                raise RuntimeError(f"request req_{i} does not match prompt {i}")
            gens.append(list(o.generated_tokens))
        gen_s = time.time() - t0

        recs, force_idx, force_seqs = [], [], []
        for k, (q, p, g) in enumerate(zip(todo, prompts, gens)):
            closed = self.think_end_id in g
            cut = g.index(self.think_end_id) if closed else len(g)
            answer = self.tok.decode(g[cut + 1:], skip_special_tokens=True) if closed else ""
            letter = parse_letter(answer, q["n_opts"]) if closed else None
            truncated = not closed and len(g) >= args.max_new_tokens
            rec = {"id": q["id"], "subject": q["subject"], "category": mmlu_category(q["subject"]), "gold": q["gold"],
                   "pred": letter, "method": "parsed" if letter else None, "truncated": truncated,
                   "think_tokens": cut, "gen_tokens": len(g), "answer": answer[:400]}
            if letter is None:
                force_idx.append(k)
                force_seqs.append(p + g[:cut] + (self.close_ids if not closed else [self.think_end_id] + self.after_close))
            recs.append(rec)
        if force_seqs:
            for k, x in zip(force_idx, self.forced_letters(force_seqs, [todo[k]["n_opts"] for k in force_idx])):
                recs[k]["pred"], recs[k]["method"] = x, "forced"
        for r in recs:
            r["correct"] = r["pred"] == r["gold"]
            self.done[r["id"]] = r
        with open(os.path.join(self.out, PRED_FILE), "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        ntok = sum(len(g) for g in gens)
        print(f"  [{self.tag}] {len(todo)} questions, {ntok:,} tokens in {gen_s:.0f}s ({ntok / gen_s:.0f} tok/s), "
              f"acc {mean(r['correct'] for r in recs):.3f}, forced {len(force_seqs)}, "
              f"truncated {sum(r['truncated'] for r in recs)}", flush=True)
        return len(todo)

    def summarize(self, n_total: int, args) -> dict:
        preds = list(self.done.values())
        nli, llm = load_ai_related(MMLU_AI_RELATED), load_ai_related(LLM_SELECTION)
        by_subj, by_cat = defaultdict(list), defaultdict(list)
        for p in preds:
            by_subj[p["subject"]].append(p["correct"])
            by_cat[p["category"]].append(p["correct"])

        def acc(f):
            xs = [p["correct"] for p in preds if f(p)]
            return round(mean(xs), 4) if xs else None

        res = {"n": len(preds), "n_total": n_total, "complete": len(preds) >= n_total, "shots": 0, "format": "chat",
               "thinking": True, "accuracy": acc(lambda p: True),
               "macro_accuracy": round(mean(mean(v) for v in by_subj.values()), 4) if by_subj else None,
               "ai_subjects_accuracy": acc(lambda p: p["subject"] in MMLU_AI_SUBJECTS),
               "ai_related_accuracy": acc(lambda p: p["id"] in nli), "llm_selected_accuracy": acc(lambda p: p["id"] in llm),
               "machine_learning_accuracy": acc(lambda p: p["subject"] == "machine_learning"),
               "by_category": {c: round(mean(v), 4) for c, v in sorted(by_cat.items())},
               "parsed_rate": round(mean(p["method"] == "parsed" for p in preds), 4) if preds else None,
               "forced_rate": round(mean(p["method"] == "forced" for p in preds), 4) if preds else None,
               "truncated_rate": round(mean(p["truncated"] for p in preds), 4) if preds else None,
               "mean_think_tokens": round(mean(p["think_tokens"] for p in preds), 1) if preds else None,
               "pred_distribution": dict(Counter(p["pred"] for p in preds)),
               "settings": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                            "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed, "order_seed": args.order_seed}}
        spath = os.path.join(self.out, "summary.json")
        summ = {}
        if os.path.exists(spath):
            with open(spath, encoding="utf-8") as f:
                summ = json.load(f)
        summ.update({"tag": self.tag, "model": self.adapter or self.base, "base": self.base})
        summ.setdefault("tasks", {})["mmlu_thinking"] = res
        summ["time_mmlu_thinking"] = time.strftime("%F %T")
        with open(spath, "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2)
        return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="TAG=MODEL[@BASE]")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order-seed", type=int, default=0, help="seed of the question order")
    ap.add_argument("--chunk", type=int, default=1024, help="questions per chunk (results are saved per chunk)")
    ap.add_argument("--max-questions", type=int, default=None, help="stop after this many (a random sample)")
    ap.add_argument("--limit-per-subject", type=int, default=None, help="smoke tests only")
    ap.add_argument("--cb-blocks", type=int, default=220,
                    help="KV-cache blocks of 256 tokens (~29 MB each for Qwen3-0.6B): caps GPU memory on a shared GPU")
    ap.add_argument("--cb-batch-tokens", type=int, default=2048, help="max tokens per continuous-batching step")
    args = ap.parse_args()

    qs = load_questions(args.split, args.order_seed, args.limit_per_subject)
    if args.max_questions:
        qs = qs[:args.max_questions]
    runners = []
    for spec in args.run:
        tag, _, model = spec.partition("=")
        runners.append(Runner(tag, model, args))
    t0 = time.time()
    for s in range(0, len(qs), args.chunk):
        chunk = qs[s:s + args.chunk]
        print(f"chunk {s // args.chunk + 1}/{(len(qs) + args.chunk - 1) // args.chunk} "
              f"(questions {s + 1}-{s + len(chunk)}), {time.time() - t0:.0f}s elapsed", flush=True)
        for r in runners:
            if r.run_chunk(chunk, args):
                res = r.summarize(len(qs), args)
                print(f"  [{r.tag}] so far n={res['n']} acc {res['accuracy']}", flush=True)
    for r in runners:
        res = r.summarize(len(qs), args)
        print(f"{r.tag:52s} n={res['n']} MMLU {res['accuracy']} AI subj {res['ai_subjects_accuracy']} "
              f"forced {res['forced_rate']} truncated {res['truncated_rate']} think {res['mean_think_tokens']}")


if __name__ == "__main__":
    main()
