#!/usr/bin/env python3
"""Aggregate cpt_data/scores/<corpus>/ outputs of score_ai_relevance.py: label shares, score distribution, per-subset table.
usage: python3 -P scripts/summarize_ai_scores.py [cpt_data/scores]  -> prints and writes <scores>/summary.txt"""
import os, sys, glob, collections
import numpy as np, pyarrow.parquet as pq
root = sys.argv[1] if len(sys.argv) > 1 else 'cpt_data/scores'
out = []
def P(*a): s = ' '.join(str(x) for x in a); print(s); out.append(s)
for corpus in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
    files = sorted(glob.glob(f'{root}/{corpus}/**/*.parquet', recursive=True))
    src = f'cpt_data/{corpus}'
    p_ai, p_near, lab, sub = [], [], [], []
    mismatch = 0
    for f in files:
        t = pq.read_table(f, columns=['p_ai', 'p_near', 'label'])
        rel = os.path.relpath(f, f'{root}/{corpus}')
        n_in = pq.ParquetFile(os.path.join(src, rel)).metadata.num_rows if os.path.exists(os.path.join(src, rel)) else -1
        if n_in != t.num_rows: mismatch += 1
        p_ai.append(t.column('p_ai').to_numpy()); p_near.append(t.column('p_near').to_numpy())
        lab += t.column('label').to_pylist(); sub += [rel.split('/')[0]] * t.num_rows
    p_ai = np.concatenate(p_ai); p_near = np.concatenate(p_near); lab = np.array(lab); sub = np.array(sub)
    n = len(p_ai)
    P(f'\n===== {corpus}: {len(files)} files, {n:,} docs, row-count mismatches vs input: {mismatch}, NaN p_ai: {int(np.isnan(p_ai).sum())} =====')
    c = collections.Counter(lab)
    P('labels:   ' + '  '.join(f'{k}={c[k]/n:.3f} ({c[k]:,})' for k in ['AI', 'near-AI', 'not-AI']))
    P(f'score (10*p_ai): mean={10*p_ai.mean():.2f}  median={10*np.median(p_ai):.2f}')
    P('p_ai thresholds: ' + '  '.join(f'>={t}: {(p_ai>=t).mean():.3f} ({int((p_ai>=t).sum()):,})' for t in [0.2, 0.5, 0.8, 0.95]))
    hist, edges = np.histogram(p_ai, bins=[0, .05, .2, .5, .8, .95, 1.0001])
    P('p_ai histogram: ' + '  '.join(f'[{edges[i]:.2f},{edges[i+1]:.2f}): {hist[i]/n:.3f}' for i in range(len(hist))))
    P(f"{'subset':26s} {'docs':>10s} {'AI':>6s} {'near':>6s} {'not':>6s} {'mean':>5s}")
    rows = []
    for s in sorted(set(sub)):
        m = sub == s; k = m.sum()
        rows.append((s, k, (lab[m] == 'AI').mean(), (lab[m] == 'near-AI').mean(), (lab[m] == 'not-AI').mean(), 10 * p_ai[m].mean()))
    for s, k, a, b, d, mu in rows:
        P(f'{s:26s} {k:>10,} {a:6.3f} {b:6.3f} {d:6.3f} {mu:5.2f}')
open(f'{root}/summary.txt', 'w').write('\n'.join(out) + '\n')
print(f'\nwritten {root}/summary.txt')
