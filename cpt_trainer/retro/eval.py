#!/usr/bin/env python3
"""Evaluate a RETRO-fitted adapter: MMLU / ai_large / AI MCQ with online retrieval, and validation loss by
retrieval mode and by neighbour-target overlap.

    python -m cpt_trainer.retro.eval --model runs/retro_qwen06b_v1_web1b/epoch1/adapter --db cpt_data/retro_db/ai_v1 \\
        --tasks mmlu,mcq,val --modes on,off,random --set runs/_retro_cache/web1b --tag retro_v1_epoch1

Modes: on = neighbours retrieved from the database; off = the decoder + LoRA alone (RETRO layers skipped);
random = random database chunks as neighbours (if `on` beats `random`, the model uses the neighbours' content).
MMLU / MCQ: scripts/eval_model.py's own prompts and log-likelihood letter scoring (5-shot, base format), through a
model wrapper that left-pads every prompt to whole chunks (so the chunk with the question and options is retrieved
for and the answer position attends to its neighbours; Wang et al. 2023 "left padding") and retrieves the
neighbours of every chunk on the fly (bge-small + faiss; results cached per chunk, so the shared 5-shot prefixes
cost one search each). The database is decontaminated against these eval sets (cpt_trainer/retro/db.py).
val (needs --set): mean token loss per validation corpus for every mode, and RETRO's leakage analysis: the tokens
of every CCA span are bucketed by the longest token run their target text shares with the chunk's neighbours
(r = 0-7 / 8-31 / >= 32 tokens); the gain of `on` over `off` in the r < 8 bucket is the gain that is not copying
(RETRO sec. 2.6, Norlund et al. 2023).
Writes eval_results/<tag>/summary.json and summary.md (+ the per-question predictions of eval_model.py per mode).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ai_qa_train.common import ROOT, load_tokenizer, read_jsonl  # noqa: E402
from cpt_trainer.retro.common import longest_common_run, row_hash  # noqa: E402
from cpt_trainer.retro.db import Database  # noqa: E402
from cpt_trainer.retro.model import RetroConfig, attach_retro, dense_document_mask, retro_hidden, unwrap  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_model", os.path.join(ROOT, "scripts", "eval_model.py"))
em = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(em)


def load_retro_model(adapter: str | None, base: str, device: str, attn: str = "sdpa"):
    """Base + RETRO module (from <adapter>/retro_config.json) + PEFT adapter (LoRA and RETRO weights)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                                                 attn_implementation=attn).to(device)
    if adapter:
        if os.path.exists(os.path.join(adapter, "retro_config.json")):
            attach_retro(model, RetroConfig.from_file(adapter), dtype=next(model.parameters()).dtype)
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


class Retriever:
    """Online retrieval for eval prompts: chunk tokens -> top-k database chunk ids (cached)."""

    def __init__(self, db: Database, k: int, nprobe: int = 32, k_search: int = 8, embedder: str | None = None):
        from cpt_trainer.retro.embed import ChunkEmbedder
        from cpt_trainer.retro.index import INDEX_NAME, load_qwen_tok, open_index
        with open(os.path.join(db.path, "index", "index.json")) as f:
            info = json.load(f)
        self.db, self.k, self.k_search = db, k, max(k_search, k)
        self.index = open_index(os.path.join(db.path, "index", INDEX_NAME), nprobe=nprobe)
        self.emb = ChunkEmbedder(os.path.join(ROOT, embedder or info["embedder"]))
        self.qtok = load_qwen_tok(db)
        self.cache: dict[bytes, np.ndarray] = {}

    def __call__(self, chunks: list[np.ndarray]) -> np.ndarray:
        """chunks: token arrays (real tokens only) -> int64 [n, k] database chunk ids (-1 = none)."""
        keys = [c.astype(np.int64).tobytes() for c in chunks]
        todo = [i for i, kk in enumerate(keys) if kk not in self.cache]
        if todo:
            texts = self.qtok.batch_decode([chunks[i].tolist() for i in todo], skip_special_tokens=True)
            _, cand = self.index.search(self.emb.embed_texts(texts), self.k_search)
            N = self.db.chunk_tokens(cand)
            h = row_hash(N)
            for j, i in enumerate(todo):
                out, seen = [], set()
                for c, hh in zip(cand[j], h[j]):
                    if c >= 0 and hh not in seen:
                        seen.add(hh); out.append(c)
                    if len(out) == self.k:
                        break
                self.cache[keys[i]] = np.asarray(out + [-1] * (self.k - len(out)), dtype=np.int64)
        return np.stack([self.cache[kk] for kk in keys]) if keys else np.zeros((0, self.k), np.int64)


class RetroScorer:
    """Stands in for the HF model in eval_model.score_letters: model(input_ids, attention_mask, logits_to_keep=1)
    -> .logits [B, 1, V] of the last real token of every (left-padded) row, computed with retrieval."""

    def __init__(self, model, retriever: Retriever | None, cfg: RetroConfig | None, pad_id: int, mode: str, seed: int = 0):
        self.model, self.retriever, self.cfg, self.pad_id, self.mode = model, retriever, cfg, pad_id, mode
        self.device = next(model.parameters()).device
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def __call__(self, input_ids, attention_mask=None, logits_to_keep=1, **kw):
        lm, _ = unwrap(self.model)
        rows = [ids[m.bool()].cpu().numpy() for ids, m in zip(input_ids, attention_mask)]
        C = self.cfg.chunk_len if self.cfg else 64
        toks, docs, pos, cpos, cend, cfirst, chunks, last = [], [], [], [], [], [], [], []
        off = 0
        for r, row in enumerate(rows):
            pad = (-len(row)) % C if self.cfg else 0
            seq = np.concatenate([np.full(pad, self.pad_id, np.int64), row])
            toks.append(seq)
            docs.append(np.concatenate([np.full(pad, -1 - r), np.full(len(row), r)]))   # padding = its own document
            pos.append(np.concatenate([np.arange(pad), np.arange(len(row))]))
            for s in range(0, len(seq) - C + 1, C):
                cpos.append(off + s); cend.append(off + len(seq)); cfirst.append(off + max(s, pad))
                chunks.append(seq[max(s, pad):s + C])
            off += len(seq)
            last.append(off - 1)
        batch = {"input_ids": torch.from_numpy(np.concatenate(toks))[None].to(self.device),
                 "position_ids": torch.from_numpy(np.concatenate(pos))[None].to(self.device)}
        doc = torch.from_numpy(np.concatenate(docs)).to(self.device)
        retrieval = self.mode != "off" and self.cfg is not None and len(chunks) > 0
        if retrieval:
            if self.mode == "random":
                ids = self.rng.integers(0, len(self.retriever.db), size=(len(chunks), self.retriever.k))
            else:
                ids = self.retriever(chunks)
            nbr = self.retriever.db.neighbours(ids, self.cfg.nbr_len)
            batch.update({"chunk_pos": torch.tensor(cpos), "chunk_end": torch.tensor(cend),
                          "chunk_first": torch.tensor(cfirst), "nbr_tokens": torch.from_numpy(nbr)})
            batch = {k: v.to(self.device) for k, v in batch.items()}
        h = retro_hidden(self.model, batch, dense_document_mask(doc), retrieval=retrieval)
        logits = (h[torch.tensor(last, device=self.device)] @ lm.lm_head.weight.t()).float()
        return SimpleNamespace(logits=logits[:, None, :])


@torch.no_grad()
def val_analysis(model, cfg, db, set_dir, nbr_name, modes, k, max_seqs, attn, log=print) -> dict:
    """Per corpus: mean loss per mode, and the loss in CCA spans bucketed by neighbour-target overlap."""
    from cpt_trainer.retro.dataset import RetroCollator, RetroSplit
    from cpt_trainer.retro.model import attention_mask_for
    lm, _ = unwrap(model)
    with open(os.path.join(set_dir, "meta.json")) as f:
        corpora = sorted(json.load(f)["stats"]["val"])
    dev = next(model.parameters()).device
    res = {}
    buckets = [(0, 8), (8, 32), (32, 10 ** 9)]
    C = cfg.chunk_len
    for c in corpora:
        ds = RetroSplit(set_dir, f"val_{c}", nbr_name, max_seqs=max_seqs)
        agg = {m: {"nll": 0.0, "n": 0, "b_nll": [0.0] * 3, "b_n": [0] * 3} for m in modes}
        for i in range(len(ds)):
            item = ds[i]
            nll_by_mode, bucket_of_token = {}, None
            for m in modes:
                col = RetroCollator(db, k, cfg.nbr_len, 0, 2048, seed=1, mode=m)
                b = {kk: v.to(dev) if torch.is_tensor(v) else v for kk, v in col([item]).items()}
                hid = retro_hidden(model, b, attention_mask_for(b["doc_ids"][0], attn), retrieval=m != "off")
                y = b["labels"][0, 1:]
                lp = torch.log_softmax((hid[:-1] @ lm.lm_head.weight.t()).float(), -1)
                nll = -lp.gather(-1, y.clamp(min=0)[:, None])[:, 0]
                nll = torch.where(y >= 0, nll, torch.zeros_like(nll)).cpu().numpy()
                nll_by_mode[m] = (nll, (y >= 0).cpu().numpy())
                if m == "on" and bucket_of_token is None:
                    # bucket of every predicted token: r = longest run shared by the span's target chunk and the neighbours
                    bucket_of_token = np.full(len(nll), -1)
                    seq = item["input_ids"].numpy()
                    nb = b["nbr_tokens"].cpu().numpy() if "nbr_tokens" in b else None
                    for j, (p0, e) in enumerate(zip(item["chunk_pos"], item["chunk_end"])):
                        s0, s1 = p0 + C - 1, min(p0 + 2 * C - 1, e)           # span positions (predicting s0+1..s1)
                        if nb is None or s1 <= s0:
                            continue
                        tgt = seq[s0 + 1:s1 + 1]
                        r = max(longest_common_run(tgt, nb[j, kk]) for kk in range(nb.shape[1]))
                        bi = next(q for q, (lo, hi) in enumerate(buckets) if lo <= r < hi)
                        bucket_of_token[s0:s1] = bi
            for m, (nll, valid) in nll_by_mode.items():
                a = agg[m]
                a["nll"] += float(nll[valid].sum()); a["n"] += int(valid.sum())
                if bucket_of_token is not None:
                    for q in range(3):
                        sel = valid & (bucket_of_token == q)
                        a["b_nll"][q] += float(nll[sel].sum()); a["b_n"][q] += int(sel.sum())
        res[c] = {m: {"loss": a["nll"] / max(1, a["n"]),
                      "by_overlap": {f"r{lo}-{hi if hi < 10 ** 9 else 'inf'}": {
                          "loss": a["b_nll"][q] / max(1, a["b_n"][q]), "tokens": a["b_n"][q]}
                          for q, (lo, hi) in enumerate(buckets)}} for m, a in agg.items()}
        log(f"  val {c}: " + ", ".join(f"{m} {v['loss']:.4f}" for m, v in res[c].items()))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="adapter dir (with retro_config.json); empty = base model")
    ap.add_argument("--base", default=os.path.join(ROOT, "models", "Qwen3-0.6B-Base"))
    ap.add_argument("--db", required=True)
    ap.add_argument("--tasks", default="mmlu,mcq")
    ap.add_argument("--modes", default="on,off")
    ap.add_argument("--k", type=int, default=2, help="neighbours per chunk at evaluation")
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--set", default=None, help="training set folder (for the val task)")
    ap.add_argument("--nbr-name", default=None)
    ap.add_argument("--max-val-seqs", type=int, default=32)
    ap.add_argument("--mmlu-subjects", default="all")
    ap.add_argument("--mmlu-max-per-subject", type=int, default=None)
    ap.add_argument("--mmlu-shots", type=int, default=5)
    ap.add_argument("--mcq-shots", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--attn", default="sdpa", help="decoder attention for the val task: sdpa (dense mask) or flex_attention")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or f"retro_{time.strftime('%m%d_%H%M')}"
    out = os.path.join(ROOT, "eval_results", tag)
    os.makedirs(out, exist_ok=True)
    tok = load_tokenizer(args.base)
    model = load_retro_model(args.model, args.base, dev)
    lm, _ = unwrap(model)
    cfg = getattr(lm, "retro_config", None)
    db = Database(os.path.abspath(args.db))
    modes = [m for m in args.modes.split(",") if m and (cfg is not None or m == "off")]
    tasks = args.tasks.split(",")
    retriever = Retriever(db, args.k, args.nprobe) if cfg is not None and ("mmlu" in tasks or "mcq" in tasks) else None
    summary = {"tag": tag, "model": args.model, "db": os.path.relpath(db.path, ROOT), "k": args.k, "modes": {}}
    t0 = time.time()
    em_args = SimpleNamespace(mmlu_dir=em.MMLU_DIR, mmlu_subjects=args.mmlu_subjects, mmlu_ai_related=em.MMLU_AI_RELATED,
                              mmlu_split="test", mmlu_shots=args.mmlu_shots, mmlu_format="base", mmlu_scoring="loglik",
                              mmlu_max_per_subject=args.mmlu_max_per_subject, mmlu_max_new_tokens=512, bs=args.bs,
                              gen_bs=16, mcq_shots=args.mcq_shots)
    for mode in modes:
        scorer = RetroScorer(model, retriever, cfg, tok.pad_token_id, mode)
        mdir = os.path.join(out, mode)
        os.makedirs(mdir, exist_ok=True)
        r = {}
        if "mmlu" in tasks:
            m = em.eval_mmlu(scorer, tok, mdir, em_args)
            r.update({k_: m.get(k_) for k_ in ("accuracy", "macro_accuracy", "ai_subjects_accuracy", "ai_related_accuracy",
                                                "ai_large_accuracy", "letter_prob_mass", "n", "warning")})
            r["by_category"] = m.get("by_category"); r["ai_large_by_topic"] = m.get("ai_large_by_topic")
            print(f"[{mode}] mmlu {r['accuracy']} ai_large {r.get('ai_large_accuracy')} ({time.time() - t0:.0f}s)", flush=True)
        if "mcq" in tasks and os.path.exists(os.path.join(em.EVAL_DIR, "mcq.jsonl")):
            rows = [x for x in read_jsonl(os.path.join(em.EVAL_DIR, "mcq.jsonl")) if x.get("ground_truth")][:args.limit]
            q = em.eval_mcq(scorer, tok, rows, mdir, em_args)
            r.update({"mcq_accuracy": q["accuracy"], "mcq_n": q["n"], "mcq_warning": q.get("warning")})
            print(f"[{mode}] mcq {q['accuracy']} ({time.time() - t0:.0f}s)", flush=True)
        summary["modes"][mode] = r
    if "val" in tasks and cfg is not None and args.set:
        summary["val"] = val_analysis(model, cfg, db, os.path.abspath(args.set), args.nbr_name or db.meta.get("name"),
                                      [m for m in ("on", "off", "random") if m in modes] or ["on", "off"], args.k,
                                      args.max_val_seqs, args.attn)
    summary["seconds"] = round(time.time() - t0)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    lines = [f"# RETRO eval `{tag}`", "", f"model `{args.model}`, database `{summary['db']}`, k={args.k}", "",
             "| mode | MMLU | macro | AI subj. | AI-rel. | ai_large | MCQ | letter mass |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    f4 = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else ""  # noqa: E731
    for mode, r in summary["modes"].items():
        lines.append(f"| {mode} | " + " | ".join(f4(r.get(k_)) for k_ in ("accuracy", "macro_accuracy", "ai_subjects_accuracy",
                     "ai_related_accuracy", "ai_large_accuracy", "mcq_accuracy", "letter_prob_mass")) + " |")
    if summary.get("val"):
        lines += ["", "## Validation loss by mode and neighbour-target overlap r (tokens)", "",
                  "| corpus | mode | all | r 0-7 | r 8-31 | r >= 32 |", "|---|---|---:|---:|---:|---:|"]
        for c, d in summary["val"].items():
            for m, v in d.items():
                b = list(v["by_overlap"].values())
                lines.append(f"| {c} | {m} | {v['loss']:.4f} | " + " | ".join(f"{x['loss']:.4f} ({x['tokens']:,})" for x in b) + " |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
