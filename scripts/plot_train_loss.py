#!/usr/bin/env python3
"""Collect the training-loss history of every run in runs/ and write docs/train_loss_curves.html (one panel per run).

Loss points come from the '| loss' step lines of train.log (CPT runs) or, when a run has none, from log_history.json
(SFT runs). Re-run it while a job is training to refresh the page.
"""
import argparse, glob, json, os, re, time

STEP_RE = re.compile(r'step (\d+)/(\d+) \| epoch ([\d.]+) \| loss ([\d.]+) \| grad_norm ([\d.]+) \| lr ([\d.e+-]+) \| ([\d.]+) M tok')


def from_train_log(path):
    pts = {}  # step -> point; a resumed run re-logs steps, keep the latest line
    total = 0
    for line in open(path, errors='replace'):
        m = STEP_RE.search(line)
        if m:
            s = int(m[1]); total = int(m[2])
            pts[s] = dict(step=s, epoch=float(m[3]), loss=float(m[4]), lr=float(m[6]), tok=float(m[7]))
    return [pts[s] for s in sorted(pts)], total


def from_log_history(path):
    pts = {}
    for h in json.load(open(path)):
        if 'loss' in h and 'step' in h:
            tok = h.get('num_input_tokens_seen')
            pts[h['step']] = dict(step=h['step'], epoch=h.get('epoch'), loss=h['loss'], lr=h.get('learning_rate'),
                                  tok=tok / 1e6 if tok else None)
    return [pts[s] for s in sorted(pts)]


def load_run(d):
    cfg = json.load(open(f'{d}/run_config.json')) if os.path.exists(f'{d}/run_config.json') else {}
    args = cfg.get('args', {})
    pts, total, src = [], 0, None
    if os.path.exists(f'{d}/train.log'):
        pts, total = from_train_log(f'{d}/train.log'); src = 'train.log'
    if not pts and os.path.exists(f'{d}/log_history.json'):
        pts = from_log_history(f'{d}/log_history.json'); src = 'log_history.json'
    if not pts:
        return None
    total = total or cfg.get('total_steps') or pts[-1]['step']
    last = pts[-1]
    log_every = args.get('logging_steps') or 10
    done = last['step'] >= total - log_every
    src_path = f'{d}/{src}'
    running = not done and time.time() - os.path.getmtime(src_path) < 30 * 60
    planned_epochs = args.get('epochs') or 1.0
    has_tok = all(p['tok'] is not None for p in pts)
    k = min(10, len(pts))
    return dict(
        name=os.path.basename(d),
        base=os.path.basename(str(args.get('base', cfg.get('base', '?')))),
        method=args.get('method', '?'), r=args.get('r'), lr=args.get('lr'),
        plan=os.path.splitext(os.path.basename(args['lora_config']))[0] if args.get('lora_config') else None,
        data=[os.path.basename(c) for c in (cfg.get('corpora') or args.get('corpora') or args.get('data') or [])],
        source=src, status='complete' if done else 'running' if running else 'stopped',
        total_steps=total, planned_epochs=planned_epochs,
        # planned token count, extrapolated from the tokens/step seen so far
        planned_tok=last['tok'] / last['step'] * total if has_tok and last['step'] else None,
        first_loss=sum(p['loss'] for p in pts[:k]) / k, final_loss=sum(p['loss'] for p in pts[-k:]) / k,
        updated=time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(src_path))),
        started=os.path.getmtime(f'{d}/run_config.json') if cfg else os.path.getmtime(src_path),
        pts=[[p['step'], p['epoch'], p['tok'] if has_tok else None, p['loss'], p['lr']] for p in pts])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--out', default='docs/train_loss_curves.html')
    a = ap.parse_args()
    runs = [r for d in sorted(glob.glob(f'{a.runs}/*/')) if not os.path.basename(d.rstrip('/')).startswith('_')
            for r in [load_run(d.rstrip('/'))] if r]
    runs.sort(key=lambda r: -r['started'])  # newest first
    tpl = open(os.path.join(os.path.dirname(__file__), 'train_loss_template.html')).read()
    html = (tpl.replace('__GENERATED__', time.strftime('%Y-%m-%d %H:%M'))
               .replace('__DATA__', json.dumps(runs, separators=(',', ':')).replace('</', '<\\/')))
    with open(a.out, 'w') as fh:
        fh.write(html)
    for r in runs:
        print(f"{r['name']:40s} {r['status']:9s} {len(r['pts']):4d} pts  loss {r['first_loss']:.3f} -> {r['final_loss']:.3f}  ({r['source']})")
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
