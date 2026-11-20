#!/usr/bin/env python3
"""Qwen3 chat model in *thinking* mode as a reading-difficulty judge (levels 1-5, rubric = difficulty_common.RUBRIC).

The model gets the rubric and the document head, reasons inside <think>...</think> and must end with "Difficulty: N".
Generation uses transformers continuous batching (generate_batch) with Qwen's recommended thinking-mode sampling
(temperature 0.6, top-p 0.95, top-k 20, fixed seed). If the reasoning hits --max-new-tokens or no level can be
parsed after </think>, the reasoning is closed and "Difficulty: " is appended; the digit 1-5 with the highest next-token
logit is taken ("forced"). Resumable: --out is a jsonl appended per chunk, rows whose idx is present are skipped.

usage: python3 -P scripts/judge_difficulty_qwen.py --input cpt_data/difficulty/sample_1k.parquet \
           --out cpt_data/difficulty/qwen17b_think.jsonl --model models/Qwen3-1.7b [--limit 8] [--chunk 100]
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import Counter

import pyarrow.parquet as pq
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from difficulty_common import RUBRIC, head, parse_level  # noqa: E402

PROMPT = (RUBRIC + "\n\nText:\n\"\"\"\n{text}\n\"\"\"\n\n"
          "Think it through briefly, then finish with exactly one line of the form:\nDifficulty: N")
THINK_END = "</think>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen3-1.7b"))
    ap.add_argument("--head", type=int, default=1200, help="chars of text shown (uses the `head` column when present)")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cb-blocks", type=int, default=1024); ap.add_argument("--cb-batch-tokens", type=int, default=4096)
    ap.add_argument("--no-think", action="store_true", help="disable thinking (plain chat answer)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from transformers.generation.configuration_utils import ContinuousBatchingConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    think_end_id = tok.convert_tokens_to_ids(THINK_END)
    digit_ids = []
    for d in "12345":
        ids = tok.encode(d, add_special_tokens=False)
        assert len(ids) == 1, (d, ids)
        digit_ids.append(ids[0])
    close_open = tok.encode("\n" + THINK_END + "\n\nDifficulty: ", add_special_tokens=False)   # reasoning not closed
    close_done = tok.encode("\n\nDifficulty: ", add_special_tokens=False)                      # after </think>

    tbl = pq.read_table(args.input)
    names = tbl.schema.names
    rows = tbl.select([c for c in ("idx", "id", "head", "text") if c in names]).to_pylist()
    for r in rows:
        if "head" not in r:
            r["head"] = head(r["text"], args.head)
    if args.limit:
        rows = rows[:args.limit]
    done = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = json.loads(line); done[p["idx"]] = p
    todo_all = [r for r in rows if r["idx"] not in done]
    print(f"{len(rows)} docs, {len(done)} done, {len(todo_all)} to judge; model {args.model}", flush=True)

    gc = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature,
                          top_p=args.top_p, top_k=args.top_k, eos_token_id=[151645, 151643], pad_token_id=tok.pad_token_id)
    cb = ContinuousBatchingConfig(num_blocks=args.cb_blocks, max_batch_tokens=args.cb_batch_tokens)

    @torch.inference_mode()
    def forced(seq: list[int]) -> int:
        x = torch.tensor([seq[-6000:]], device=model.device)
        logits = model(input_ids=x).logits[0, -1].float()
        return 1 + int(torch.argmax(logits[digit_ids]))

    t_all = time.time()
    for s in range(0, len(todo_all), args.chunk):
        todo = todo_all[s:s + args.chunk]
        prompts = []
        for r in todo:
            text = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(text=r["head"])}],
                                           add_generation_prompt=True, enable_thinking=not args.no_think, tokenize=False)
            prompts.append(tok(text, add_special_tokens=False)["input_ids"])
        torch.manual_seed(args.seed)
        t0 = time.time()
        res = model.generate_batch(inputs=prompts, generation_config=gc, continuous_batching_config=cb)
        gens = []
        for i, p in enumerate(prompts):
            o = res[f"req_{i}"]
            if list(o.prompt_ids) != p:
                raise RuntimeError(f"request req_{i} does not match prompt {i}")
            gens.append(list(o.generated_tokens))
        gen_s = time.time() - t0
        recs = []
        for r, p, g in zip(todo, prompts, gens):
            closed = think_end_id in g
            cut = g.index(think_end_id) if closed else len(g)
            if args.no_think:
                closed, cut = True, -1
            answer = tok.decode(g[cut + 1:], skip_special_tokens=True).strip() if closed else ""
            level = parse_level(answer) if closed else None
            method = "parsed" if level else "forced"
            if level is None:
                seq = p + g[:cut] + (close_done if closed and not args.no_think else (g if args.no_think else close_open))
                if args.no_think:
                    seq = p + g + close_done
                level = forced(seq)
            think_text = tok.decode(g[:cut], skip_special_tokens=True) if not args.no_think else ""
            recs.append({"idx": r["idx"], "id": r["id"], "level": level, "method": method,
                         "truncated": (not closed) and len(g) >= args.max_new_tokens,
                         "think_tokens": cut if not args.no_think else 0, "gen_tokens": len(g),
                         "answer": answer[:300], "think_tail": think_text[-300:]})
        with open(args.out, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in recs)
        ntok = sum(len(g) for g in gens)
        print(f"  chunk {s // args.chunk + 1}: {len(todo)} docs, {ntok:,} tokens in {gen_s:.0f}s ({ntok / max(gen_s, 1e-9):.0f} tok/s), "
              f"levels {dict(sorted(Counter(x['level'] for x in recs).items()))}, forced {sum(x['method'] == 'forced' for x in recs)}, "
              f"truncated {sum(x['truncated'] for x in recs)}, mean think {sum(x['think_tokens'] for x in recs) / len(recs):.0f} tok, "
              f"elapsed {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
