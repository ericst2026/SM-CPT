#!/usr/bin/env python3
"""Merge cpt_data/scores/*.token_counts.json into one per-label table (per corpus + combined) and append to summary.txt."""
import json, glob, os
LABELS = ['AI', 'near-AI', 'not-AI']
res = {os.path.basename(f).replace('.token_counts.json', ''): json.load(open(f)) for f in sorted(glob.glob('cpt_data/scores/*.token_counts.json'))}
lines = []
def P(s=''): print(s); lines.append(s)
def table(title, d):
    tt = sum(d[k]['tokens'] for k in LABELS); td = sum(d[k]['docs'] for k in LABELS)
    P(f'\n== {title}: {td:,} docs, {tt/1e9:.2f} B Qwen3 tokens ==')
    P(f"{'label':8s} {'docs':>12s} {'doc%':>6s} {'Qwen3 tokens':>16s} {'B tok':>6s} {'tok%':>6s} {'tok/doc':>8s}")
    for k in LABELS:
        o = d[k]; P(f"{k:8s} {o['docs']:>12,} {100*o['docs']/td:6.1f} {o['tokens']:>16,} {o['tokens']/1e9:6.2f} {100*o['tokens']/tt:6.1f} {o['tokens']/max(o['docs'],1):8.0f}")
comb = {k: {'docs': 0, 'tokens': 0} for k in LABELS}
for name, r in res.items():
    table(name, r['total'])
    for k in LABELS:
        comb[k]['docs'] += r['total'][k]['docs']; comb[k]['tokens'] += r['total'][k]['tokens']
if len(res) > 1: table('BOTH CORPORA', comb)
for name, r in res.items():
    P(f'\n-- {name} per subset: B Qwen3 tokens (AI / near-AI / not-AI) --')
    for sub, d in sorted(r['per_subset'].items()):
        P(f"{sub:26s} " + ' / '.join(f"{d[k]['tokens']/1e9:5.2f}" for k in LABELS) + f"   AI tok% {100*d['AI']['tokens']/max(sum(d[k]['tokens'] for k in LABELS),1):5.1f}")
with open('cpt_data/scores/summary.txt', 'a') as fh: fh.write('\n\n===== Qwen3 token counts per label =====\n' + '\n'.join(lines) + '\n')
print('\nappended to cpt_data/scores/summary.txt')
