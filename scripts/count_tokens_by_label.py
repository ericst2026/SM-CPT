#!/usr/bin/env python3
"""Exact Qwen3-tokenizer token counts per AI-relevance label (AI / near-AI / not-AI) for a scored corpus.
Joins cpt_data/<corpus>/<f>.parquet (text) with cpt_data/scores/<corpus>/<f>.parquet (label) by position after
checking the id columns match. Parallel over files. Also reports the upstream `token_count` column when present.
usage: python3 -P scripts/count_tokens_by_label.py --corpus cpt_data/corpus_ai_fineweb_large [--workers 16] [--limit-files N]
       -> prints a table and writes cpt_data/scores/<corpus>.token_counts.json
"""
import argparse, os, sys, glob, json, time, collections
from multiprocessing import Pool
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument('--corpus', required=True); ap.add_argument('--scores', default=None)
ap.add_argument('--tokenizer', default='models/Qwen3-0.6B-Base'); ap.add_argument('--workers', type=int, default=16)
ap.add_argument('--limit-files', type=int, default=None)
a = ap.parse_args()
name = os.path.basename(a.corpus.rstrip('/')); scores_dir = a.scores or os.path.join('cpt_data', 'scores', name)
LABELS = ['AI', 'near-AI', 'not-AI']
_tok = None
def init():
    global _tok
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(a.tokenizer).backend_tokenizer
def work(f):
    rel = os.path.relpath(f, a.corpus); sub = rel.split('/')[0]
    src = pq.read_table(f, columns=['id', 'text'] + (['token_count'] if 'token_count' in pq.ParquetFile(f).schema_arrow.names else []))
    sc = pq.read_table(os.path.join(scores_dir, rel), columns=['id', 'label'])
    assert src.num_rows == sc.num_rows and src.column('id').to_pylist() == sc.column('id').to_pylist(), f'id mismatch in {rel}'
    labels = sc.column('label').to_pylist(); texts = src.column('text').to_pylist()
    up = src.column('token_count').to_pylist() if 'token_count' in src.column_names else None
    out = {k: {'docs': 0, 'tokens': 0, 'chars': 0, 'upstream_tokens': 0} for k in LABELS}
    B = 512
    for s in range(0, len(texts), B):
        encs = _tok.encode_batch(texts[s:s + B], add_special_tokens=False)
        for j, e in enumerate(encs):
            i = s + j; o = out[labels[i]]
            o['docs'] += 1; o['tokens'] += len(e.ids); o['chars'] += len(texts[i])
            if up is not None: o['upstream_tokens'] += up[i] or 0
    return sub, out
files = sorted(glob.glob(os.path.join(a.corpus, '**', '*.parquet'), recursive=True))
files = [f for f in files if '/.git/' not in f]
if a.limit_files: files = files[:a.limit_files]
t0 = time.time(); total = collections.defaultdict(lambda: collections.defaultdict(int)); per_sub = {}
with Pool(a.workers, initializer=init) as pool:
    for i, (sub, out) in enumerate(pool.imap_unordered(work, files)):
        d = per_sub.setdefault(sub, {k: collections.defaultdict(int) for k in LABELS})
        for k in LABELS:
            for m, v in out[k].items(): d[k][m] += v; total[k][m] += v
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            done = sum(total[k]['tokens'] for k in LABELS)
            print(f'{i+1}/{len(files)} files  {done/1e9:.2f} B tokens  {time.time()-t0:.0f}s', file=sys.stderr, flush=True)
def table(title, d):
    tot_tok = sum(d[k]['tokens'] for k in LABELS); tot_docs = sum(d[k]['docs'] for k in LABELS)
    print(f'\n== {title}: {tot_docs:,} docs, {tot_tok/1e9:.3f} B Qwen3 tokens ==')
    print(f"{'label':8s} {'docs':>12s} {'doc%':>6s} {'Qwen3 tokens':>14s} {'tok%':>6s} {'tok/doc':>8s} {'chars/tok':>9s}" + (f" {'upstream tok':>13s}" if d['AI']['upstream_tokens'] else ''))
    for k in LABELS:
        o = d[k]
        print(f"{k:8s} {o['docs']:>12,} {100*o['docs']/max(tot_docs,1):6.1f} {o['tokens']:>14,} {100*o['tokens']/max(tot_tok,1):6.1f} {o['tokens']/max(o['docs'],1):8.0f} {o['chars']/max(o['tokens'],1):9.2f}" + (f" {o['upstream_tokens']:>13,}" if o['upstream_tokens'] else ''))
table(name, total)
for sub in sorted(per_sub): table(f'{name}/{sub}', per_sub[sub])
json.dump({'corpus': name, 'files': len(files), 'total': total, 'per_subset': per_sub}, open(os.path.join('cpt_data', 'scores', f'{name}.token_counts.json'), 'w'), indent=1, default=dict)
print(f'\nwritten cpt_data/scores/{name}.token_counts.json  ({time.time()-t0:.0f}s)', file=sys.stderr)
