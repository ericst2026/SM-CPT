"""Evaluation report: a self-contained HTML dashboard per run + an index of all runs, served locally.

    build_report(out_dir, summary, compare)  -> eval_results/<tag>/report.html
    build_index(results_root)                -> eval_results/index.html (all runs, headline metrics)
    serve(results_root, page, host, port)    -> http://host:port/<tag>/report.html (blocks until Ctrl-C)

The page has no external dependencies (inline CSS/JS, data embedded as JSON) so it can also be opened
directly from disk or copied elsewhere.
"""
from __future__ import annotations

import glob
import json
import os
import random

AREA_FILES = {
    "mmlu": "mmlu_predictions.jsonl",
    "mcq": "mcq_predictions.jsonl",
    "open_qa": "open_qa_predictions.jsonl",
    "code": "code_predictions.jsonl",
    "math": "math_predictions.jsonl",
}
AREA_TITLES = {"mmlu": "MMLU", "mcq": "AI multiple choice", "open_qa": "Open QA by learner level",
               "code": "DS-1000 code", "math": "Math"}


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _ok(area, r):
    if area == "code":
        return bool(r.get("passed"))
    if area == "open_qa":
        return bool(r.get("level_fit")) if r.get("judged_match") is None else bool(r.get("judged_match"))
    return bool(r.get("correct"))


def _group(area, r):
    return r.get("subject") or r.get("level") or r.get("library") or r.get("source") or "all"


def sample_examples(out_dir: str, per_area: int, seed: int = 0) -> dict:
    """Balanced sample per area: spread over groups (subjects / levels / libraries), mixing right and wrong."""
    rng = random.Random(seed)
    out = {}
    for area, fname in AREA_FILES.items():
        rows = _read(os.path.join(out_dir, fname))
        if not rows:
            continue
        by_g: dict[str, list] = {}
        for r in rows:
            r["_ok"] = _ok(area, r)
            r["_group"] = _group(area, r)
            by_g.setdefault(r["_group"], []).append(r)
        quota = max(2, per_area // max(1, len(by_g)))
        picked = []
        for g, rs in by_g.items():
            good = [r for r in rs if r["_ok"]]
            bad = [r for r in rs if not r["_ok"]]
            rng.shuffle(good)
            rng.shuffle(bad)
            half = quota // 2
            take = bad[:half] + good[:quota - min(half, len(bad))]
            picked += take[:quota]
        rng.shuffle(picked)
        out[area] = picked[:max(per_area, len(by_g) * 2)]
    return out


def _json_for_script(obj) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\!--")


def build_report(out_dir: str, summary: dict, compare: dict | None = None, per_area: int = 300) -> str:
    data = {"summary": summary, "compare": compare, "examples": sample_examples(out_dir, per_area),
            "areaTitles": AREA_TITLES}
    html = PAGE.replace("__TITLE__", f"Eval report · {summary.get('tag', '')}").replace("__DATA__", _json_for_script(data))
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# --------------------------------------------------------------------------- #
def headline(summary: dict) -> dict:
    t = summary.get("tasks", {})
    g = lambda *p: _dig(t, p)  # noqa: E731
    return {
        "MMLU": g("mmlu", "accuracy"), "MMLU AI subjects": g("mmlu", "ai_subjects_accuracy"),
        "MMLU ai_related": g("mmlu", "ai_related_accuracy"), "ai_large": g("mmlu", "ai_large_accuracy"),
        "AI MCQ": g("mcq", "accuracy"), "Open QA ROUGE-L": g("open_qa", "rougeL"),
        "Open QA level fit": g("open_qa", "level_fit"), "DS-1000 pass@1": g("code", "pass@1"),
        "Math": g("math", "accuracy"), "Val loss": g("val_loss", "loss"),
    }


def _dig(d, path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def build_index(results_root: str) -> str:
    runs = []
    for p in sorted(glob.glob(os.path.join(results_root, "*", "summary.json"))):
        with open(p) as f:
            s = json.load(f)
        runs.append({"tag": os.path.basename(os.path.dirname(p)), "time": s.get("time"), "model": s.get("model"),
                     "metrics": headline(s), "has_report": os.path.exists(os.path.join(os.path.dirname(p), "report.html"))})
    runs.sort(key=lambda r: r["time"] or "", reverse=True)
    html = INDEX.replace("__DATA__", _json_for_script(runs))
    path = os.path.join(results_root, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def serve(results_root: str, page: str, host: str = "127.0.0.1", port: int = 7870, open_browser: bool = True):
    import functools
    import http.server
    import webbrowser

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):  # keep the console clean
            pass

    handler = functools.partial(Quiet, directory=results_root)
    for p in range(port, port + 20):
        try:
            httpd = http.server.ThreadingHTTPServer((host, p), handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"no free port in {port}..{port + 19}")
    url = f"http://{host}:{p}/{page}"
    print(f"\nEvaluation report: {url}\nAll runs:          http://{host}:{p}/index.html\n(Ctrl-C to stop the server)", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------- #
# Shared CSS (palette: validated 2-slot categorical, light #2a78d6/#eb6834, dark #3987e5/#d95926)
# --------------------------------------------------------------------------- #
CSS = r"""
:root{color-scheme:light;
 --page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--band:#e1e0d9;
 --good:#0ca30c;--goodText:#006300;--bad:#d03b3b;--warn:#fab219;--chip:#f0efec;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926;--band:#383835;--goodText:#0ca30c;--chip:#262624;}}
:root[data-theme="dark"]{color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926;--band:#383835;--goodText:#0ca30c;--chip:#262624;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px;font-weight:600}
h2{font-size:17px;margin:32px 0 12px;font-weight:600}
h3{font-size:14px;margin:0;font-weight:600}
.sub{color:var(--ink2);font-size:13px}
.muted{color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:16px;margin-bottom:16px;min-width:0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:16px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}
.tile .lab{color:var(--ink2);font-size:12px}
.tile .val{font-size:26px;font-weight:600;margin-top:2px}
.delta{font-size:12px;font-variant-numeric:tabular-nums}
.delta.up{color:var(--goodText)} .delta.down{color:var(--bad)}
.cardhead{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:10px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:2px 0 8px}
.key{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
button.lnk{background:none;border:1px solid var(--ring);color:var(--ink2);border-radius:6px;padding:3px 8px;font-size:12px;cursor:pointer;white-space:nowrap}
button.lnk:hover{color:var(--ink)}
.chart{position:relative}
.row{display:grid;grid-template-columns:minmax(90px,190px) 1fr;gap:10px;align-items:center}
.rlabel{color:var(--ink2);font-size:12px;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.track{padding-right:58px;min-width:0}
.plot{position:relative;display:flex;flex-direction:column;gap:2px;padding:4px 0;border-right:1px solid var(--grid);
 background-image:repeating-linear-gradient(to right,var(--grid) 0 1px,transparent 1px 25%)}
.bar{height:14px;border-radius:0 4px 4px 0;min-width:1px;position:relative}
.bar .v{position:absolute;left:calc(100% + 6px);top:-1px;font-size:11px;color:var(--ink);white-space:nowrap;font-variant-numeric:tabular-nums}
.bar.s2 .v{color:var(--ink2)}
.axis{display:grid;grid-template-columns:minmax(90px,190px) 1fr;gap:10px;font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
.axis .ticks{position:relative;height:16px;margin-right:58px}
.axis .ticks span{position:absolute;transform:translateX(-50%)}
.hit{cursor:default}
.hit:hover .rlabel{color:var(--ink)}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;
 padding:8px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:10;max-width:280px}
table.t{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
table.t th,table.t td{padding:4px 6px;border-bottom:1px solid var(--grid);text-align:right}
table.t th:first-child,table.t td:first-child{text-align:left}
.tablewrap{overflow-x:auto;margin-top:8px}
.range{position:relative;height:18px}
.range .bandr{position:absolute;top:4px;height:10px;background:var(--band);border-radius:3px}
.range .dot{position:absolute;top:3px;width:12px;height:12px;border-radius:50%;background:var(--s1);box-shadow:0 0 0 2px var(--surface);transform:translateX(-50%)}
.range .dot.s2{background:var(--s2)}
/* examples */
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.tab{border:1px solid var(--ring);background:var(--surface);color:var(--ink2);border-radius:999px;padding:5px 12px;cursor:pointer;font-size:13px}
.tab[aria-selected="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;align-items:center}
select,input[type=search]{background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:6px;padding:5px 8px;font:inherit;font-size:13px;max-width:100%}
input[type=search]{min-width:180px;flex:1}
.ex{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px;margin-bottom:10px;min-width:0}
.exhead{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
.chip{background:var(--chip);color:var(--ink2);border-radius:999px;padding:1px 8px;font-size:11px}
.status{font-size:12px;font-weight:600;display:inline-flex;align-items:center;gap:4px}
.status.ok{color:var(--goodText)} .status.no{color:var(--bad)}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:10px 0 3px}
.txt{white-space:pre-wrap;word-break:break-word}
pre{white-space:pre-wrap;word-break:break-word;background:var(--chip);border-radius:6px;padding:8px;margin:0;font-size:12px;max-height:360px;overflow:auto}
.choices{margin:4px 0 0;padding:0;list-style:none}
.choices li{padding:3px 6px;border-radius:5px;margin:2px 0}
.choices li.gold{outline:1px solid var(--good)}
.choices li.pred{background:var(--chip)}
details summary{cursor:pointer;color:var(--ink2);font-size:12px;margin-top:8px}
.more{display:block;margin:8px auto}
.theme{position:absolute;top:24px;right:16px}
header{position:relative}
@media (max-width:560px){.row,.axis{grid-template-columns:90px 1fr}.tile .val{font-size:22px}}
"""

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>""" + CSS + r"""</style></head><body>
<main>
<header><h1 id="title"></h1><div class="sub" id="meta"></div>
<button class="lnk theme" id="themeBtn" title="Toggle light/dark">◐ theme</button></header>
<div class="tiles" id="tiles"></div>
<h2>Scores</h2>
<div id="charts"></div>
<h2>Examples</h2>
<div class="tabs" id="tabs" role="tablist"></div>
<div class="filters" id="filters"></div>
<div id="exlist"></div>
<p class="muted" style="margin-top:32px;font-size:12px"><a href="../index.html" style="color:inherit">← all evaluation runs</a></p>
</main>
<div id="tip"></div>
<script>
const D = __DATA__;
const S = D.summary, C = D.compare, T = S.tasks || {}, CT = (C && C.tasks) || {};
const $ = (s, el=document) => el.querySelector(s);
const h = (tag, attrs={}, ...kids) => { const e = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs)) { if (k==='class') e.className=v; else if (k==='text') e.textContent=v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v); }
  for (const k of kids) if (k!=null) e.append(k); return e; };
const dig = (o, ...p) => p.reduce((a,k) => (a && typeof a==='object') ? a[k] : undefined, o);
const pct = v => v==null ? '–' : (v*100).toFixed(1) + '%';
const num = (v, d=3) => v==null ? '–' : (+v).toFixed(d);

// theme toggle (OS default; button forces the other)
$('#themeBtn').onclick = () => { const r=document.documentElement; const dark = r.dataset.theme ? r.dataset.theme==='dark' : matchMedia('(prefers-color-scheme: dark)').matches; r.dataset.theme = dark ? 'light' : 'dark'; };

$('#title').textContent = 'Evaluation · ' + S.tag;
$('#meta').textContent = `model ${S.model} · ${S.time || ''} · thinking ${S.thinking ? 'on' : 'off'}` + (C ? ` · compared with ${C.tag}` : '');

// ---------- tooltip ----------
const tip = $('#tip');
function showTip(ev, html) { tip.replaceChildren(...html); tip.style.display='block'; moveTip(ev); }
function moveTip(ev) { const w = tip.offsetWidth, x = Math.min(ev.clientX + 14, innerWidth - w - 8); tip.style.left = x+'px'; tip.style.top = (ev.clientY + 14)+'px'; }
function hideTip() { tip.style.display='none'; }

// ---------- stat tiles ----------
function tile(label, v, cv, fmt, higherBetter=true) {
  if (v == null) return null;
  const t = h('div', {class:'tile'}, h('div',{class:'lab',text:label}), h('div',{class:'val',text:fmt(v)}));
  if (cv != null) { const d = v - cv, good = higherBetter ? d >= 0 : d <= 0;
    const s = (d>=0?'+':'') + (fmt===pct ? (d*100).toFixed(1)+' pts' : d.toFixed(3));
    t.append(h('div',{class:'delta ' + (Math.abs(d)<1e-9 ? '' : (good?'up':'down')), text:`${good?'▲':'▼'} ${s} vs ${C.tag}`})); }
  return t;
}
const tiles = [
  tile('MMLU accuracy', dig(T,'mmlu','accuracy'), dig(CT,'mmlu','accuracy'), pct),
  tile('MMLU · AI subjects', dig(T,'mmlu','ai_subjects_accuracy'), dig(CT,'mmlu','ai_subjects_accuracy'), pct),
  tile('MMLU · ai_related', dig(T,'mmlu','ai_related_accuracy'), dig(CT,'mmlu','ai_related_accuracy'), pct),
  tile('ai_large (1k ML / AI)', dig(T,'mmlu','ai_large_accuracy'), dig(CT,'mmlu','ai_large_accuracy'), pct),
  tile('AI multiple choice', dig(T,'mcq','accuracy'), dig(CT,'mcq','accuracy'), pct),
  tile('Open QA level fit', dig(T,'open_qa','level_fit'), dig(CT,'open_qa','level_fit'), pct),
  tile('Open QA ROUGE-L', dig(T,'open_qa','rougeL'), dig(CT,'open_qa','rougeL'), pct),
  tile('DS-1000 pass@1', dig(T,'code','pass@1'), dig(CT,'code','pass@1'), pct),
  tile('Math accuracy', dig(T,'math','accuracy'), dig(CT,'math','accuracy'), pct),
  tile('Validation loss', dig(T,'val_loss','loss'), dig(CT,'val_loss','loss'), v=>num(v,3), false),
].filter(Boolean);
$('#tiles').append(...tiles);
// degenerate-scoring warnings (e.g. chat prompt format on a base model)
for (const [name, t] of Object.entries(T)) if (t && t.warning)
  $('#tiles').before(h('div',{class:'card',role:'alert',style:'border-color:var(--bad);margin-top:16px'},
    h('span',{class:'status no',text:'⚠ Warning · ' + name}), h('div',{class:'txt',style:'margin-top:4px',text:t.warning})));

// ---------- bar chart ----------
// rows: [{label, v, cv, n}] ; opts: {max, fmt, title, subtitle, higherBetter}
function barCard(title, subtitle, rows, opts={}) {
  rows = rows.filter(r => r.v != null);
  if (!rows.length) return null;
  const fmt = opts.fmt || pct, max = opts.max || Math.max(1e-9, ...rows.map(r => Math.max(r.v, r.cv ?? 0))) ;
  const hasC = rows.some(r => r.cv != null);
  const card = h('div', {class:'card'});
  const tableBtn = h('button', {class:'lnk', text:'Table', 'aria-expanded':'false'});
  card.append(h('div',{class:'cardhead'}, h('div',{}, h('h3',{text:title}), subtitle ? h('div',{class:'sub',text:subtitle}) : null), tableBtn));
  if (hasC) card.append(h('div',{class:'legend'},
      h('span',{}, h('span',{class:'key',style:'background:var(--s1)'}), S.tag),
      h('span',{}, h('span',{class:'key',style:'background:var(--s2)'}), C.tag)));
  const chart = h('div', {class:'chart'});
  const ticks = [0, .25, .5, .75, 1].map(f => f*max);   // gridlines are the .plot background at 25 % steps
  for (const r of rows) {
    const plot = h('div',{class:'plot'}), track = h('div',{class:'track'}, plot);
    const b1 = h('div',{class:'bar', style:`width:${Math.max(0.3, r.v/max*100)}%;background:var(--s1)`}, h('span',{class:'v',text:fmt(r.v)}));
    plot.append(b1);
    if (hasC && r.cv != null) plot.append(h('div',{class:'bar s2', style:`width:${Math.max(0.3, r.cv/max*100)}%;background:var(--s2)`}, h('span',{class:'v',text:fmt(r.cv)})));
    const row = h('div',{class:'row hit'}, h('div',{class:'rlabel',text:r.label,title:r.label}), track);
    row.addEventListener('mousemove', ev => { const lines=[h('div',{style:'font-weight:600;margin-bottom:2px',text:r.label}), h('div',{text:`${S.tag}: ${fmt(r.v)}`})];
      if (r.cv != null) { const d=r.v-r.cv; lines.push(h('div',{text:`${C.tag}: ${fmt(r.cv)}`}), h('div',{class:'muted',text:`Δ ${d>=0?'+':''}${fmt===pct?(d*100).toFixed(1)+' pts':d.toFixed(3)}`})); }
      if (r.n != null) lines.push(h('div',{class:'muted',text:`n = ${r.n}`}));
      if (r.extra) lines.push(h('div',{class:'muted',text:r.extra}));
      showTip(ev, lines); });
    row.addEventListener('mouseleave', hideTip);
    chart.append(row);
  }
  const ax = h('div',{class:'ticks'}); for (const t of ticks) ax.append(h('span',{style:`left:${t/max*100}%`,text:fmt(t)}));
  chart.append(h('div',{class:'axis'}, h('div'), ax));
  card.append(chart);
  // table view
  const tbl = h('table',{class:'t'}); tbl.append(h('tr',{}, h('th',{text:'item'}), h('th',{text:S.tag}), hasC ? h('th',{text:C.tag}) : null, hasC ? h('th',{text:'Δ'}) : null, h('th',{text:'n'})));
  for (const r of rows) tbl.append(h('tr',{}, h('td',{text:r.label}), h('td',{text:fmt(r.v)}), hasC ? h('td',{text:r.cv==null?'–':fmt(r.cv)}) : null,
      hasC ? h('td',{text:r.cv==null?'–':(fmt===pct?((r.v-r.cv)*100).toFixed(1):(r.v-r.cv).toFixed(3))}) : null, h('td',{text:r.n ?? ''})));
  const wrap = h('div',{class:'tablewrap',hidden:''}, tbl); card.append(wrap);
  tableBtn.onclick = () => { const open = wrap.hidden; wrap.hidden = !open; tableBtn.setAttribute('aria-expanded', open); tableBtn.textContent = open ? 'Chart only' : 'Table'; };
  return card;
}

const charts = $('#charts');
function addRow(...cards) { cards = cards.filter(Boolean); if (!cards.length) return; const g = h('div',{class: cards.length>1 ? 'grid2' : ''}); g.append(...cards); charts.append(g); }

// overview
const ov = [
  ['MMLU', 'mmlu','accuracy'], ['MMLU macro', 'mmlu','macro_accuracy'], ['MMLU AI subjects', 'mmlu','ai_subjects_accuracy'],
  ['MMLU ai_related', 'mmlu','ai_related_accuracy'], ['ai_large', 'mmlu','ai_large_accuracy'],
  ['AI MCQ', 'mcq','accuracy'], ['Open QA ROUGE-L', 'open_qa','rougeL'], ['Open QA token-F1', 'open_qa','token_f1'],
  ['Open QA level fit', 'open_qa','level_fit'], ['Open QA judged level', 'open_qa','judged_match'],
  ['DS-1000 pass@1', 'code','pass@1'], ['DS-1000 (env-valid)', 'code','pass@1_valid'], ['Math', 'math','accuracy'],
].map(([label, t, k]) => ({label, v: dig(T,t,k), cv: dig(CT,t,k), n: dig(T,t,'n')}));
addRow(barCard('All scores', 'Every headline metric, 0–100 %', ov, {max:1}));

if (T.mmlu) {
  const m = T.mmlu;
  addRow(barCard('MMLU by category', `${m.split} split · ${m.shots}-shot · ${m.n} questions`,
    Object.entries(m.by_category).map(([k,v]) => ({label:k, v:v.acc, cv:dig(CT,'mmlu','by_category',k,'acc'), n:v.n})), {max:1}));
  const subj = Object.entries(m.by_subject).map(([k,v]) => ({label:k.replaceAll('_',' '), v:v.acc, cv:dig(CT,'mmlu','by_subject',k,'acc'), n:v.n, extra:v.category}))
     .sort((a,b) => b.v - a.v);
  addRow(barCard('MMLU by subject', 'Sorted by accuracy', subj, {max:1}));
}
if (T.mcq) addRow(barCard('AI multiple choice by subject', `${T.mcq.n} questions`,
  Object.entries(T.mcq.by_subject).map(([k,v]) => ({label:k.replaceAll('_',' '), v:v.acc, cv:dig(CT,'mcq','by_subject',k,'acc'), n:v.n})).sort((a,b)=>b.v-a.v), {max:1}));
if (T.open_qa) {
  const pl = T.open_qa.per_level, cpl = dig(CT,'open_qa','per_level') || {};
  const rowsFor = k => Object.entries(pl).map(([lv,v]) => ({label:lv.replaceAll('_',' '), v:v[k], cv:dig(cpl,lv,k), n:v.n}));
  addRow(barCard('Open QA · ROUGE-L by level', 'Overlap with the reference answer', rowsFor('rougeL'), {max:Math.max(.5, ...rowsFor('rougeL').map(r=>Math.max(r.v||0,r.cv||0)))}),
         barCard('Open QA · level fit by level', 'Share of answers whose reading grade fits the audience', rowsFor('level_fit'), {max:1}));
  if (T.open_qa.judged_match != null) addRow(barCard('Open QA · judged audience match', 'Zero-shot classifier agrees the answer targets the requested level', rowsFor('judged_match'), {max:1}));
  // reading-grade vs target range
  const LV = {primary_school:[0,5.5],middle_school:[5.5,8.5],high_school:[8.5,12.5],undergrad_non_ai:[12.5,15],ai_student:[12.5,18],general_public:[6,12],researcher_non_ai:[13,20],ai_researcher:[15,30]};
  const card = h('div',{class:'card'}, h('div',{class:'cardhead'}, h('div',{}, h('h3',{text:'Reading grade vs target range'}),
     h('div',{class:'sub',text:'Gray band = target Flesch-Kincaid grade for the level; dot = mean grade of the model\'s answers'}))));
  if (C) card.append(h('div',{class:'legend'}, h('span',{}, h('span',{class:'key',style:'background:var(--s1);border-radius:50%'}), S.tag), h('span',{}, h('span',{class:'key',style:'background:var(--s2);border-radius:50%'}), C.tag)));
  const MAXG = 24, chart = h('div',{class:'chart'});
  for (const [lv, v] of Object.entries(pl)) {
    const [lo, hi] = LV[lv] || [0,0]; const rg = h('div',{class:'range'});
    rg.append(h('div',{class:'bandr',style:`left:${lo/MAXG*100}%;width:${(Math.min(hi,MAXG)-lo)/MAXG*100}%`}));
    const cv = dig(cpl, lv, 'fk_grade');
    if (C && cv != null) rg.append(h('div',{class:'dot s2',style:`left:${Math.min(cv,MAXG)/MAXG*100}%`}));
    if (v.fk_grade != null) rg.append(h('div',{class:'dot',style:`left:${Math.min(v.fk_grade,MAXG)/MAXG*100}%`}));
    const row = h('div',{class:'row hit'}, h('div',{class:'rlabel',text:lv.replaceAll('_',' ')}), h('div',{class:'track'}, h('div',{class:'plot'}, rg)));
    row.addEventListener('mousemove', ev => showTip(ev, [h('div',{style:'font-weight:600',text:lv}), h('div',{text:`target grade ${lo}–${hi}`}),
       h('div',{text:`${S.tag}: grade ${num(v.fk_grade,1)}`}), C && cv!=null ? h('div',{text:`${C.tag}: grade ${num(cv,1)}`}) : null, h('div',{class:'muted',text:`mean answer length ${num(v.answer_words,0)} words`})].filter(Boolean)));
    row.addEventListener('mouseleave', hideTip); chart.append(row);
  }
  const ax = h('div',{class:'ticks'}); for (const t of [0,6,12,18,24]) ax.append(h('span',{style:`left:${t/MAXG*100}%`,text:'grade '+t}));
  chart.append(h('div',{class:'axis'}, h('div'), ax)); card.append(chart); addRow(card);
}
if (T.code) addRow(barCard('DS-1000 pass@1 by library', `${T.code.n} problems · reference solutions pass ${pct(T.code.reference_pass_rate)} in this environment`,
  Object.entries(T.code.by_library).map(([k,v]) => ({label:k, v:v['pass@1'], cv:dig(CT,'code','by_library',k,'pass@1'), n:v.n, extra:`reference pass ${pct(v.reference_pass)}`})).sort((a,b)=>b.v-a.v), {max:1}));
if (T.val_loss && T.val_loss.per_level_loss) addRow(barCard('Validation loss by level', 'Assistant-token cross-entropy · lower is better',
  Object.entries(T.val_loss.per_level_loss).map(([k,v]) => ({label:k.replaceAll('_',' '), v, cv:dig(CT,'val_loss','per_level_loss',k)})), {fmt:v=>num(v,3)}));

// ---------- examples ----------
const EX = D.examples || {}, areas = Object.keys(EX);
let cur = areas.includes(location.hash.slice(1)) ? location.hash.slice(1) : areas[0], shown = 0; const PAGE = 30;
const tabs = $('#tabs');
for (const a of areas) { const b = h('button',{class:'tab',role:'tab','aria-selected':String(a===cur),text:`${D.areaTitles[a]||a} (${EX[a].length})`});
  b.onclick = () => { cur = a; history.replaceState(null, '', '#' + a); for (const x of tabs.children) x.setAttribute('aria-selected', String(x===b)); buildFilters(); render(); };
  tabs.append(b); }
if (!areas.length) $('#exlist').append(h('p',{class:'muted',text:'No prediction files found.'}));
const fStatus = h('select',{}, ...['all','correct','wrong'].map(v => h('option',{value:v,text:v==='all'?'All results':v==='correct'?'Correct / fitting':'Wrong / not fitting'})));
const fGroup = h('select',{}), fSearch = h('input',{type:'search',placeholder:'Search question or answer…'});
$('#filters').append(fStatus, fGroup, fSearch);
for (const el of [fStatus, fGroup]) el.onchange = render; fSearch.oninput = render;
function buildFilters() { const gs = [...new Set((EX[cur]||[]).map(r => r._group))].sort();
  fGroup.replaceChildren(h('option',{value:'',text:'All groups'}), ...gs.map(g => h('option',{value:g,text:g.replaceAll('_',' ')}))); }
function filtered() { const s = fSearch.value.toLowerCase(), st = fStatus.value, g = fGroup.value;
  return (EX[cur]||[]).filter(r => (st==='all' || (st==='correct') === r._ok) && (!g || r._group===g) &&
    (!s || JSON.stringify([r.question, r.answer, r.solution, r.output, r.reference]).toLowerCase().includes(s))); }
function status(ok, yes='Correct', no='Wrong') { return h('span',{class:'status '+(ok?'ok':'no'),text:(ok?'✓ ':'✗ ')+(ok?yes:no)}); }
const L = (t) => h('div',{class:'lbl',text:t});
const add = (el, ...kids) => { for (const k of kids) if (k != null) el.append(k); return el; };
function card(r) {
  const e = h('div',{class:'ex'}), head = h('div',{class:'exhead'});
  if (cur==='mmlu') {
    add(head, status(r._ok), h('span',{class:'chip',text:r.subject.replaceAll('_',' ')}), h('span',{class:'chip',text:r.category}));
    const ul = h('ul',{class:'choices'}); (r.choices||[]).forEach((c,i) => { const L_='ABCDEFGHIJ'[i];
      const li = h('li',{class:(L_===r.gold?'gold ':'')+(L_===r.pred?'pred':'')}, `${L_}. ${c}`);
      if (L_===r.gold) li.append(h('span',{class:'muted',text:'  ← correct'})); if (L_===r.pred) li.append(h('span',{class:'muted',text:'  ← model'})); ul.append(li); });
    e.append(head, h('div',{class:'txt',text:r.question}), ul);
    if (r.output != null) {  // generative scoring: the reply the letter was parsed from
      if (!r.pred) head.append(h('span',{class:'chip',text:'no answer letter found'}));
      e.append(h('details',{}, h('summary',{text:'Model reply' + (r.parse ? ` (parsed: ${r.parse.replaceAll('_',' ')})` : '')}), h('pre',{text:r.output})));
    }
  } else if (cur==='mcq') {
    add(head, status(r._ok), h('span',{class:'chip',text:r.subject}), h('span',{class:'chip',text:r.source}));
    e.append(head, h('div',{class:'txt',text:r.question}), L('Model'), h('div',{text:r.pred}), L('Correct'), h('div',{text:r.answer || r.gold}));
  } else if (cur==='open_qa') {
    add(head, status(r._ok, r.judged_match!=null?'Audience matched':'Level fit', r.judged_match!=null?'Audience mismatch':'Grade out of range'),
      h('span',{class:'chip',text:r.level.replaceAll('_',' ')}), h('span',{class:'chip',text:`ROUGE-L ${num(r.rougeL,2)}`}),
      h('span',{class:'chip',text:`grade ${num(r.fk_grade,1)}`}), r.judged_level ? h('span',{class:'chip',text:`judged: ${r.judged_level}`}) : null);
    e.append(head, h('div',{class:'txt',style:'font-weight:500',text:r.question}), L('Model answer'), h('div',{class:'txt',text:r.answer}));
    if (r.thinking) e.append(h('details',{}, h('summary',{text:'Reasoning trace'}), h('pre',{text:r.thinking})));
    e.append(h('details',{}, h('summary',{text:'Reference answer'}), h('div',{class:'txt',text:r.reference||''})));
  } else if (cur==='code') {
    add(head, status(r._ok,'Passed','Failed'), h('span',{class:'chip',text:r.library}), r.reference_passes===false ? h('span',{class:'chip',text:'reference fails in this env'}) : null);
    e.append(head, h('details',{}, h('summary',{text:'Problem'}), h('pre',{text:r.question||''})), L('Model solution'), h('pre',{text:r.solution||''}));
    if (!r._ok && r.error) e.append(L('Error'), h('div',{class:'txt muted',text:r.error}));
    if (r.reference) e.append(h('details',{}, h('summary',{text:'Reference solution'}), h('pre',{text:r.reference})));
  } else {
    add(head, status(r._ok)); e.append(head, h('div',{class:'txt',text:r.question||''}), L('Model output'), h('pre',{text:r.output||''}), L('Ground truth'), h('div',{text:r.gold}));
  }
  return e;
}
function render(more) { const list = $('#exlist'), rows = filtered(); if (!more) { list.replaceChildren(); shown = 0; }
  $('.more', list)?.remove();
  for (const r of rows.slice(shown, shown + PAGE)) list.append(card(r)); shown = Math.min(rows.length, shown + PAGE);
  if (!rows.length) list.append(h('p',{class:'muted',text:'No examples match the filters.'}));
  if (shown < rows.length) list.append(h('button',{class:'lnk more',text:`Show more (${rows.length-shown} left)`,onclick:()=>render(true)})); }
if (areas.length) { buildFilters(); render(); }
</script></body></html>
"""

INDEX = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Evaluation runs</title><style>""" + CSS + r"""
td a{color:var(--s1)}</style></head><body><main>
<h1>Evaluation runs</h1><div class="sub">Every run in eval_results/, newest first. Open a run for charts and examples.</div>
<div class="card" style="margin-top:16px"><div class="tablewrap"><table class="t" id="t"></table></div></div></main>
<script>
const R = __DATA__;
const t = document.getElementById('t');
const keys = R.length ? Object.keys(R[0].metrics) : [];
const tr = document.createElement('tr'); for (const k of ['run','time', ...keys]) { const th=document.createElement('th'); th.textContent=k; tr.append(th); } t.append(tr);
for (const r of R) { const row = document.createElement('tr');
  const a = document.createElement(r.has_report ? 'a' : 'span'); a.textContent = r.tag; if (r.has_report) a.href = r.tag + '/report.html';
  const td0 = document.createElement('td'); td0.append(a); row.append(td0);
  const td1 = document.createElement('td'); td1.textContent = r.time || ''; row.append(td1);
  for (const k of keys) { const v = r.metrics[k]; const td=document.createElement('td');
    td.textContent = v==null ? '–' : (k==='Val loss' ? v.toFixed(3) : (v*100).toFixed(1)+'%'); row.append(td); }
  t.append(row); }
</script></body></html>
"""
