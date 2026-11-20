#!/usr/bin/env python3
"""Compare the reading-difficulty judges on the 1k sample against Claude's hand labels (the reference).

Inputs
  --sample   cpt_data/difficulty/sample_1k.parquet            (idx, corpus, ai_label, head, text, n_tokens, ...)
  --auto     cpt_data/difficulty/sample_1k.auto.parquet       (classify_difficulty.py --templates all: zs + length)
  --qwen     cpt_data/difficulty/qwen17b_think.jsonl          (judge_difficulty_qwen.py; may be partial)  [repeatable NAME=path]
  --claude   cpt_data/difficulty/claude                       (labels_NN.jsonl = reference, labels_overlap.jsonl = re-label)
Outputs
  --out-md   cpt_data/difficulty/report.md ; --out-parquet cpt_data/difficulty/sample_1k.labels.parquet (all labels joined)

Metrics vs reference: exact accuracy, within-one-level accuracy, quadratic-weighted kappa (QWK), Spearman, MAE, bias
(mean predicted - mean reference). "calibrated" rows map a continuous score to levels with quantile-matched thresholds
(predicted level distribution := reference distribution) fitted on even idx and applied to odd idx and vice versa
(2-fold CV); thresholds fitted on all 1000 are printed for use in classify_difficulty.py.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from collections import Counter

import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from difficulty_common import LENGTH_FEATURES, LEVEL_TITLE, ZS_TEMPLATES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="cpt_data/difficulty/sample_1k.parquet")
ap.add_argument("--auto", default="cpt_data/difficulty/sample_1k.auto.parquet")
ap.add_argument("--qwen", action="append", default=[], help="NAME=path.jsonl (repeatable)")
ap.add_argument("--claude", default="cpt_data/difficulty/claude")
ap.add_argument("--out-md", default="cpt_data/difficulty/report.md")
ap.add_argument("--out-parquet", default="cpt_data/difficulty/sample_1k.labels.parquet")
a = ap.parse_args()
if not a.qwen:
    a.qwen = ["qwen17b_think=cpt_data/difficulty/qwen17b_think.jsonl", "qwen17b_nothink=cpt_data/difficulty/qwen17b_nothink.jsonl"]

L = []  # report lines
def P(s=""): L.append(s)

# ------------------------------------------------------------------ load & join
df = pq.read_table(a.sample).to_pandas().set_index("idx").sort_index()
auto = pq.read_table(a.auto).to_pandas()
assert list(auto["id"]) == list(df["id"]), "auto file rows do not match the sample"
auto.index = df.index
df = df.join(auto.drop(columns=[c for c in auto.columns if c in df.columns or c == "id"]))

ref = {}
for f in sorted(glob.glob(os.path.join(a.claude, "labels_[0-9][0-9].jsonl"))):
    for line in open(f, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); ref[int(r["idx"])] = int(r["level"])
df["claude"] = pd.Series(ref)
ovl = {}
fo = os.path.join(a.claude, "labels_overlap.jsonl")
if os.path.exists(fo):
    for line in open(fo, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); ovl[int(r["idx"])] = int(r["level"])
df["claude_relabel"] = pd.Series(ovl)

qwen_meta = {}
for spec in a.qwen:
    name, _, path = spec.partition("=")
    if not os.path.exists(path):
        print(f"missing {path}, skipping {name}", file=sys.stderr); continue
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    df[name] = pd.Series({int(r["idx"]): int(r["level"]) for r in recs})
    qwen_meta[name] = {"n": len(recs), "forced": sum(r["method"] == "forced" for r in recs),
                       "truncated": sum(bool(r.get("truncated")) for r in recs),
                       "mean_think": float(np.mean([r["think_tokens"] for r in recs])) if recs else 0.0,
                       "mean_gen": float(np.mean([r["gen_tokens"] for r in recs])) if recs else 0.0}

has_ref = df["claude"].notna()
n_ref = int(has_ref.sum())


# ------------------------------------------------------------------ metrics
def metrics(ref_s: pd.Series, pred_s: pd.Series) -> dict | None:
    m = ref_s.notna() & pred_s.notna()
    if m.sum() < 10:
        return None
    r, p = ref_s[m].astype(int).values, pred_s[m].astype(int).values
    return {"n": int(m.sum()), "acc": float((r == p).mean()), "acc±1": float((np.abs(r - p) <= 1).mean()),
            "qwk": float(cohen_kappa_score(r, p, weights="quadratic", labels=[1, 2, 3, 4, 5])),
            "spearman": float(spearmanr(r, p).correlation), "mae": float(np.abs(r - p).mean()),
            "bias": float(p.mean() - r.mean()), "mean_pred": float(p.mean())}


def fmt_row(name, m):
    if m is None:
        return f"| {name} | – | | | | | | |"
    return (f"| {name} | {m['n']} | {m['acc']:.3f} | {m['acc±1']:.3f} | {m['qwk']:.3f} | {m['spearman']:.3f} | "
            f"{m['mae']:.2f} | {m['bias']:+.2f} |")


HDR = "| method | n | acc | acc ±1 | QWK | Spearman | MAE | bias |\n|---|---|---|---|---|---|---|---|"


def quantile_thresholds(score: np.ndarray, ref_levels: np.ndarray) -> list[float]:
    """Thresholds on `score` such that the predicted level histogram equals the reference histogram."""
    cum = np.cumsum(np.bincount(ref_levels, minlength=6)[1:5]) / len(ref_levels)   # P(level<=1..4)
    return [float(np.quantile(score, q)) for q in cum]


def apply_thresholds(score, th):
    return 1 + np.searchsorted(np.array(th), score, side="right")


def calibrated(score_col: str) -> tuple[pd.Series, list[float]]:
    """2-fold CV quantile calibration of a continuous column against the reference; returns CV predictions + full-fit thresholds."""
    s = df[score_col].astype(float)
    pred = pd.Series(np.nan, index=df.index)
    folds = [df.index[(df.index % 2 == 0) & has_ref], df.index[(df.index % 2 == 1) & has_ref]]
    for fit_idx, app_idx in ((folds[0], folds[1]), (folds[1], folds[0])):
        th = quantile_thresholds(s[fit_idx].values, df.loc[fit_idx, "claude"].astype(int).values)
        pred[app_idx] = apply_thresholds(s[app_idx].values, th)
    th_all = quantile_thresholds(s[has_ref].values, df.loc[has_ref, "claude"].astype(int).values)
    return pred, th_all


# ------------------------------------------------------------------ report
P("# Reading-difficulty judges on 1,000 sampled documents")
P()
P(f"Sample: {len(df)} docs, {dict(Counter(df['corpus']))}; AI labels {dict(Counter(df['ai_label']))}. "
  f"Reference = Claude labels on the 1,200-char head ({n_ref} docs). Levels: " +
  ", ".join(f"{i}={LEVEL_TITLE[i]}" for i in range(1, 6)) + ".")
P()
P("## Reference label distribution (Claude)")
P()
P("| slice | n | 1 | 2 | 3 | 4 | 5 | mean |\n|---|---|---|---|---|---|---|---|")
def dist_row(name, s):
    s = s.dropna().astype(int)
    c = Counter(s)
    return f"| {name} | {len(s)} | " + " | ".join(f"{c.get(i, 0) / max(len(s), 1):.2f}" for i in range(1, 6)) + f" | {s.mean():.2f} |"
P(dist_row("all", df["claude"]))
for c in sorted(df["corpus"].unique()):
    P(dist_row(c, df.loc[df["corpus"] == c, "claude"]))
for c in ("AI", "near-AI", "not-AI"):
    P(dist_row(f"ai_label={c}", df.loc[df["ai_label"] == c, "claude"]))
for c in sorted(df.loc[df["corpus"] == "corpus_ai_nemotron_large", "subset"].unique()):
    P(dist_row(f"nemotron/{c}", df.loc[df["subset"] == c, "claude"]))
P()
m_self = metrics(df["claude"], df["claude_relabel"])
if m_self:
    P(f"Claude self-consistency (independent re-label of {m_self['n']} docs): acc {m_self['acc']:.3f}, "
      f"acc±1 {m_self['acc±1']:.3f}, QWK {m_self['qwk']:.3f}, Spearman {m_self['spearman']:.3f}, MAE {m_self['mae']:.2f}. "
      "This is the ceiling any other judge can be expected to reach against this reference.")
P()

P("## Judges vs the reference")
P()
P(HDR)
rows = []
if m_self:
    rows.append(("Claude re-label (self-consistency ceiling)", m_self))
for name in qwen_meta:
    rows.append((f"Qwen3-1.7B {name.split('_', 1)[1]} (LLM judge)", metrics(df["claude"], df[name])))
for T in ZS_TEMPLATES:
    rows.append((f"zero-shot DeBERTa `{T}` argmax", metrics(df["claude"], df[f"{T}_level"])))
calib_th = {}
TEMPS = (1.0, 2.0, 4.0, 8.0, 16.0)
for T in ZS_TEMPLATES:
    lo = df[[f"{T}_lo{i}" for i in range(1, 6)]].astype(float).values
    best = None
    for temp in TEMPS:
        pr = np.exp(lo / temp); pr /= pr.sum(1, keepdims=True)
        df[f"{T}_score_t{temp:g}"] = (pr * np.arange(1, 6)).sum(1)
        pred, th = calibrated(f"{T}_score_t{temp:g}"); calib_th[f"{T}_score_t{temp:g}"] = th
        m = metrics(df["claude"], pred)
        rows.append((f"zero-shot DeBERTa `{T}` expected-level T={temp:g}, calibrated (CV)", m))
        if best is None or m["qwk"] > best[0]:
            best = (m["qwk"], temp, pred)
    df[f"{T}_level_cal"] = best[2]; df[f"{T}_score"] = df[f"{T}_score_t{best[1]:g}"]
    calib_th[f"{T}_score"] = calib_th[f"{T}_score_t{best[1]:g}"]
    print(f"{T}: best temperature {best[1]:g} (QWK {best[0]:.3f})", file=sys.stderr)
for col, name in (("len_level_fk", "length: Flesch-Kincaid grade, fixed thresholds"),
                  ("len_level_sent", "length: avg sentence length, fixed thresholds"),
                  ("len_level_max", "length: max sentence length, fixed thresholds"),
                  ("len_level_tok", "length: document tokens, fixed thresholds")):
    if col in df:
        rows.append((name, metrics(df["claude"], df[col])))
for col in ("fk_grade", "avg_sent_len", "max_sent_len", "p90_sent_len", "avg_word_len", "long_word_ratio", "n_tokens"):
    pred, th = calibrated(col); calib_th[col] = th
    df[f"{col}_level_cal"] = pred
    rows.append((f"length: {col}, calibrated thresholds (CV)", metrics(df["claude"], pred)))


def combo(features: list[str], alpha: float = 1.0):
    """Ridge regression of the reference level on `features` (2-fold CV on even/odd idx) + quantile calibration.
    Returns CV level predictions and the full-fit model in raw-feature space for difficulty_common.COMBO."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = df[features].astype(float).fillna(0.0)
    if "n_tokens" in features:
        X["n_tokens"] = np.log1p(X["n_tokens"])
    X = X.values; y = df["claude"].values
    pred = pd.Series(np.nan, index=df.index)
    pos = {i: k for k, i in enumerate(df.index)}
    folds = [np.array([pos[i] for i in df.index if i % 2 == par and has_ref[i]]) for par in (0, 1)]
    for fit, app in ((folds[0], folds[1]), (folds[1], folds[0])):
        mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[fit], y[fit].astype(float))
        th = quantile_thresholds(mdl.predict(X[fit]), y[fit].astype(int))
        pred.iloc[app] = apply_thresholds(mdl.predict(X[app]), th)
    allr = np.concatenate(folds)
    mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[allr], y[allr].astype(float))
    sc, rg = mdl.named_steps["standardscaler"], mdl.named_steps["ridge"]
    coef = rg.coef_ / sc.scale_
    spec = {"features": features, "coef": [round(float(c), 6) for c in coef],
            "intercept": round(float(rg.intercept_ - np.sum(coef * sc.mean_)), 6),
            "thresholds": [round(float(t), 4) for t in quantile_thresholds(mdl.predict(X[allr]), y[allr].astype(int))],
            "log1p": ["n_tokens"] if "n_tokens" in features else []}
    return pred, spec


best_T = max(ZS_TEMPLATES, key=lambda T: (metrics(df["claude"], df[f"{T}_level_cal"]) or {"qwk": -1})["qwk"])
combo_specs = {}
for name, feats in (("combo: best zero-shot log-odds + length features (ridge, CV)",
                     [f"{best_T}_lo{i}" for i in range(1, 6)] + ["fk_grade", "long_word_ratio", "avg_word_len", "avg_sent_len", "n_tokens"]),
                    ("combo: all three zero-shot templates' log-odds + length features (ridge, CV)",
                     [f"{T}_lo{i}" for T in ZS_TEMPLATES for i in range(1, 6)] + ["fk_grade", "long_word_ratio", "avg_word_len", "avg_sent_len", "n_tokens"]),
                    ("combo: length features only (ridge, CV)",
                     ["fk_grade", "long_word_ratio", "avg_word_len", "avg_sent_len", "max_sent_len", "n_tokens"])):
    pred, spec = combo(feats)
    df[name] = pred; combo_specs[name] = spec
    rows.append((name, metrics(df["claude"], pred)))

for name, m in rows:
    P(fmt_row(name, m))
P()
P("acc = exact level agreement; acc ±1 = within one level; QWK = quadratic-weighted Cohen's kappa (1 = perfect, 0 = chance); "
  "bias = mean predicted level minus mean reference level. Calibrated rows use quantile-matched thresholds fitted on the other half of the sample.")
P()

P("## Raw score correlation with the reference level")
P()
P("| score | Spearman | Pearson |\n|---|---|---|")
cands = [f"{T}_score" for T in ZS_TEMPLATES] + list(qwen_meta) + ["fk_grade", "avg_sent_len", "max_sent_len", "p90_sent_len",
         "avg_word_len", "long_word_ratio", "n_tokens", "n_words", "p_ai", "fw_edu_score", "kw_core"]
for c in cands:
    if c not in df:
        continue
    m = has_ref & df[c].notna()
    if m.sum() < 10:
        continue
    x, y = df.loc[m, c].astype(float).values, df.loc[m, "claude"].astype(int).values
    P(f"| {c} | {spearmanr(x, y).correlation:.3f} | {pearsonr(x, y)[0]:.3f} |")
P()

P("## Confusion matrices (rows = Claude reference 1..5, columns = predicted 1..5)")
P()
best_zs = max(ZS_TEMPLATES, key=lambda T: (metrics(df["claude"], df[f"{T}_level"]) or {"qwk": -1})["qwk"])
best_zs_cal = max(ZS_TEMPLATES, key=lambda T: (metrics(df["claude"], df[f"{T}_level_cal"]) or {"qwk": -1})["qwk"])
len_cols = [c for c in df if c.endswith("_level_cal") and not any(c.startswith(T) for T in ZS_TEMPLATES)]
best_len = max(len_cols, key=lambda c: (metrics(df["claude"], df[c]) or {"qwk": -1})["qwk"])
best_combo = max(combo_specs, key=lambda c: (metrics(df["claude"], df[c]) or {"qwk": -1})["qwk"])
show = [(n, n) for n in qwen_meta] + [(f"zero-shot `{best_zs}` argmax", f"{best_zs}_level"),
        (f"zero-shot `{best_zs_cal}` calibrated", f"{best_zs_cal}_level_cal"), (f"length {best_len}", best_len), (best_combo, best_combo)]
if m_self:
    show.insert(0, ("Claude re-label", "claude_relabel"))
for name, col in show:
    m = has_ref & df[col].notna()
    if m.sum() < 10:
        continue
    cm = confusion_matrix(df.loc[m, "claude"].astype(int), df.loc[m, col].astype(int), labels=[1, 2, 3, 4, 5])
    P(f"**{name}** (n={int(m.sum())})")
    P()
    P("| ref \\ pred | 1 | 2 | 3 | 4 | 5 |\n|---|---|---|---|---|---|")
    for i in range(5):
        P(f"| {i + 1} | " + " | ".join(str(int(v)) for v in cm[i]) + " |")
    P()

P("## Agreement between the automatic judges (no reference needed)")
P()
P(HDR.replace("method", "pair (first = treated as reference)"))
pairs = [(q, f"{best_zs}_level") for q in qwen_meta] + [(q, f"{best_zs_cal}_level_cal") for q in qwen_meta] + \
        [(q, best_len) for q in qwen_meta] + [(f"{best_zs}_level", best_len)]
if len(qwen_meta) == 2:
    pairs.insert(0, tuple(qwen_meta))
for x, y in pairs:
    P(fmt_row(f"{x} vs {y}", metrics(df[x], df[y])))
P()

P("## Slices (QWK / exact acc vs the reference)")
P()
slice_cols = list(qwen_meta) + [f"{best_zs}_level", f"{best_zs_cal}_level_cal", best_len, best_combo]
P("| slice | n | " + " | ".join(slice_cols) + " |\n|---|---|" + "---|" * len(slice_cols))
slices = [("all", has_ref)] + [(c, df["corpus"] == c) for c in sorted(df["corpus"].unique())] + \
         [(f"ai_label={c}", df["ai_label"] == c) for c in ("AI", "near-AI", "not-AI")]
for name, mask in slices:
    cells = []
    for c in slice_cols:
        m = metrics(df.loc[mask, "claude"], df.loc[mask, c])
        cells.append(f"{m['qwk']:.2f} / {m['acc']:.2f}" if m else "–")
    P(f"| {name} | {int((mask & has_ref).sum())} | " + " | ".join(cells) + " |")
P()

if qwen_meta:
    P("## Qwen judge statistics")
    P()
    P("| run | n | forced | truncated | mean think tokens | mean generated tokens |\n|---|---|---|---|---|---|")
    for name, q in qwen_meta.items():
        P(f"| {name} | {q['n']} | {q['forced']} | {q['truncated']} | {q['mean_think']:.0f} | {q['mean_gen']:.0f} |")
    P()

P("## Calibrated thresholds fitted on all reference docs (level = 1 + number of thresholds the score reaches)")
P()
P("| score | t1→2 | t2→3 | t3→4 | t4→5 |\n|---|---|---|---|---|")
for c, th in calib_th.items():
    P(f"| {c} | " + " | ".join(f"{t:.3f}" for t in th) + " |")
P()

P("## Ridge combo models fitted on all reference docs (difficulty_score = intercept + sum coef*feature; n_tokens enters as log1p)")
P()
for name, spec in combo_specs.items():
    P(f"**{name}**")
    P()
    P("```json\n" + json.dumps(spec) + "\n```")
    P()

P("## Largest disagreements with the reference (document heads, first 160 chars)")
P()
for name, col in [(n, n) for n in list(qwen_meta)[:1]] + [(f"zero-shot `{best_zs}`", f"{best_zs}_level")]:
    m = has_ref & df[col].notna()
    d = (df.loc[m, col].astype(int) - df.loc[m, "claude"].astype(int))
    worst = d.abs().sort_values(ascending=False).head(8).index
    P(f"**{name}**")
    P()
    for i in worst:
        P(f"- idx {i} ({df.at[i, 'corpus'].replace('corpus_ai_', '')}, {df.at[i, 'ai_label']}): ref {int(df.at[i, 'claude'])}, "
          f"pred {int(df.at[i, col])}: `{df.at[i, 'head'][:160]}`")
    P()

os.makedirs(os.path.dirname(a.out_md) or ".", exist_ok=True)
open(a.out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
keep = [c for c in df.columns if c != "text"]
df[keep].reset_index().to_parquet(a.out_parquet, index=False)
print("\n".join(L))
print(f"\nwrote {a.out_md} and {a.out_parquet}", file=sys.stderr)
