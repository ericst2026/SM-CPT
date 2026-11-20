"""Zero-shot NLI classifier on top of models/deberta-v3-base-zeroshot-v2.0.

The model is an NLI cross-encoder: (premise=text, hypothesis=label description)
-> [entailment, not_entailment]. P(entailment) is used as the label score.

* multi-label: every hypothesis is scored independently (sigmoid-like, sums != 1)
* single-label: entailment logits are softmaxed across hypotheses

Implementation notes
  - We call the model directly instead of `pipeline("zero-shot-classification")`:
    it lets us batch (text x hypothesis) pairs, sort by length and run fp16,
    which is ~3-5x faster on large corpora.
  - Texts are truncated to `max_length` tokens (premise side only).
"""
from __future__ import annotations

import os
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "deberta-v3-base-zeroshot-v2.0",
)


class ZeroShotClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = 64, max_length: int = 256, fp16: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.float16 if (fp16 and self.device.startswith("cuda")) else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, dtype=dtype)
        self.model.to(self.device).eval()
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.entail_id = label2id.get("entailment", 0)
        self.batch_size = batch_size
        self.max_length = max_length

    @torch.inference_mode()
    def _entail_logits(self, pairs: Sequence[tuple[str, str]], max_length: int | None = None) -> list[float]:
        """Return entailment-vs-not log-odds for each (premise, hypothesis) pair."""
        max_length = max_length or self.max_length
        out = [0.0] * len(pairs)
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))  # length bucketing
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            enc = self.tok(
                [pairs[i][0] for i in idx], [pairs[i][1] for i in idx],
                truncation="only_first", max_length=max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.float()
            other = 1 - self.entail_id
            lo = (logits[:, self.entail_id] - logits[:, other]).tolist()
            for i, v in zip(idx, lo):
                out[i] = v
        return out

    def multi_label(self, texts: Sequence[str], hypotheses: dict[str, str],
                    max_length: int | None = None) -> list[dict[str, float]]:
        """Independent P(entailment) per label."""
        names = list(hypotheses)
        pairs = [(t, hypotheses[n]) for t in texts for n in names]
        lo = torch.tensor(self._entail_logits(pairs, max_length)).view(len(texts), len(names))
        probs = torch.sigmoid(lo)
        return [{n: round(float(p), 4) for n, p in zip(names, row)} for row in probs]

    def single_label(self, texts: Sequence[str], hypotheses: dict[str, str],
                    max_length: int | None = None) -> list[dict[str, float]]:
        """Softmax over labels (exactly one label is assumed to be true)."""
        names = list(hypotheses)
        pairs = [(t, hypotheses[n]) for t in texts for n in names]
        lo = torch.tensor(self._entail_logits(pairs, max_length)).view(len(texts), len(names))
        probs = lo.softmax(-1)
        return [{n: round(float(p), 4) for n, p in zip(names, row)} for row in probs]


def chunked(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
