#!/usr/bin/env python3
"""Batched GPU zero-shot AI-relevance scorer for the cpt_data parquet corpora.

Model: models/deberta-v3-base-zeroshot-v2.0 (NLI, labels entailment / not_entailment).
For each document the first HEAD characters of `text` are the premise and two hypotheses are tested:
    p_ai   = P(entailment | "This text is about artificial intelligence, machine learning, neural networks, or data science.")
    p_near = P(entailment | "This text is about computing, software, data, electronics, or robotics.")
    score  = 10 * p_ai                         (0-10, comparable to the manual 0-10 rating)
    label  = "AI"      if p_ai   >= 0.5
             "near-AI" if p_ai   <  0.5 and p_near >= 0.5
             "not-AI"  otherwise
Output: one parquet per input parquet under --out, same relative path, columns id, p_ai, p_near, score, label.
Validated on 200 hand-scored docs (2026-10-30): pearson 0.78 vs manual 0-10; label AI at p_ai>=0.5 gives acc 0.88,
precision 0.84, recall 0.77 against manual>=5 (threshold 0.2 trades to P 0.82 / R 0.84). Tokens beyond the first
1000 chars are never seen; 600-char heads lose ~0.05 correlation.
Resumable: an output file whose row count matches its input is skipped. Safe to re-run after a kill.

usage:
  python3 -P scripts/score_ai_relevance.py --corpus cpt_data/corpus_ai_fineweb_large --out cpt_data/scores/corpus_ai_fineweb_large
  options: --head 1000 --max-len 384 --batch 128 --limit-files N --near/--no-near
"""
import argparse, os, sys, time, json
import pyarrow as pa, pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HYP_AI = "This text is about artificial intelligence, machine learning, neural networks, or data science."
HYP_NEAR = "This text is about computing, software, data, electronics, or robotics."

ap = argparse.ArgumentParser()
ap.add_argument('--corpus', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--model', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'deberta-v3-base-zeroshot-v2.0'))
ap.add_argument('--head', type=int, default=1000); ap.add_argument('--max-len', type=int, default=384)
ap.add_argument('--batch', type=int, default=128); ap.add_argument('--limit-files', type=int, default=None)
ap.add_argument('--near', dest='near', action='store_true', default=True); ap.add_argument('--no-near', dest='near', action='store_false')
ap.add_argument('--device', default='cuda')
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSequenceClassification.from_pretrained(a.model, dtype=torch.float16).to(a.device).eval()
ENT = model.config.label2id['entailment']

files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(a.corpus) if '/.git' not in dp for f in fs if f.endswith('.parquet'))
if a.limit_files: files = files[:a.limit_files]
total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
print(f'{len(files)} files, {total_rows:,} docs, near={a.near}', file=sys.stderr, flush=True)

@torch.no_grad()
def entail(premises, hyp):
    out = torch.empty(len(premises))
    order = sorted(range(len(premises)), key=lambda i: len(premises[i]))   # length-sorted batches, less padding
    for s in range(0, len(order), a.batch):
        idx = order[s:s + a.batch]
        enc = tok([premises[i] for i in idx], [hyp] * len(idx), truncation='only_first', max_length=a.max_len,
                  padding=True, return_tensors='pt').to(a.device)
        logits = model(**enc).logits.float()
        out[idx] = torch.softmax(logits, -1)[:, ENT].cpu()
    return out

t0 = time.time(); done_rows = 0; scored_rows = 0; skipped = 0
for fi, f in enumerate(files):
    rel = os.path.relpath(f, a.corpus); of = os.path.join(a.out, rel)
    n_in = pq.ParquetFile(f).metadata.num_rows
    if os.path.exists(of):
        try:
            if pq.ParquetFile(of).metadata.num_rows == n_in:
                skipped += 1; done_rows += n_in; continue
        except Exception:
            pass
    tbl = pq.read_table(f, columns=['id', 'text'])
    ids = tbl.column('id').to_pylist()
    heads = [' '.join((t or '').split())[:a.head] for t in tbl.column('text').to_pylist()]
    del tbl
    p_ai = entail(heads, HYP_AI)
    p_near = entail(heads, HYP_NEAR) if a.near else torch.full((len(heads),), float('nan'))
    label = ['AI' if x >= 0.5 else ('near-AI' if (a.near and y >= 0.5) else 'not-AI') for x, y in zip(p_ai.tolist(), p_near.tolist())]
    os.makedirs(os.path.dirname(of), exist_ok=True)
    tmp = of + '.tmp'
    pq.write_table(pa.table({'id': ids, 'p_ai': p_ai.numpy(), 'p_near': p_near.numpy(), 'score': (10 * p_ai).numpy(), 'label': label}), tmp)
    os.replace(tmp, of)
    done_rows += len(ids); scored_rows += len(ids)
    el = time.time() - t0; rate = scored_rows / el if el else 0
    n_ai = sum(l == 'AI' for l in label); n_near = sum(l == 'near-AI' for l in label)
    print(f'[{fi+1}/{len(files)}] {rel}: {len(ids)} docs  AI={n_ai/len(ids):.2f} near={n_near/len(ids):.2f}  '
          f'cum {done_rows:,}/{total_rows:,}  {rate:.0f} docs/s  ETA {(total_rows-done_rows)/max(rate,1e-9)/3600:.1f} h', file=sys.stderr, flush=True)
print(f'done: {scored_rows:,} docs scored this run, {skipped} files ({done_rows-scored_rows:,} docs) skipped as already scored, {(time.time()-t0)/3600:.2f} h', file=sys.stderr)
