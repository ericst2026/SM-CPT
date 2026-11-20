"""Shared helpers for training / evaluation / chat with Qwen3 + LoRA adapters.

* load_model()            base model (+ optional 4-bit) (+ optional LoRA adapter)
* build_lora_config()     LoRA family: lora, qlora, dora, qdora, rslora, pissa, olora, loraplus
* encode_chat()           Qwen3 chat template + assistant-only labels (multi-turn aware)
* generate()              batched generation with thinking on/off
* split_thinking()        '<think>..</think> answer' -> (thinking, answer)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE = os.path.join(ROOT, "models", "Qwen3-1.7b")
IGNORE = -100

# --------------------------------------------------------------------------- #
# LoRA variants
# --------------------------------------------------------------------------- #
LORA_METHODS = {
    "lora":     "standard LoRA (bf16 base)",
    "qlora":    "LoRA on a 4-bit NF4 quantised base (bitsandbytes) - lowest memory",
    "dora":     "weight-decomposed LoRA (magnitude + direction) - often better quality, ~20-30% slower",
    "qdora":    "DoRA on a 4-bit base",
    "rslora":   "rank-stabilised LoRA (scale = alpha/sqrt(r)) - better for large ranks",
    "pissa":    "LoRA initialised from the principal singular vectors of W - faster convergence",
    "olora":    "LoRA initialised with an orthonormal (QR) basis",
    "loraplus": "LoRA with a larger learning rate for B than for A (LoRA+, ratio --loraplus-ratio)",
}

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TARGET_PRESETS = {
    "all-linear": ALL_LINEAR,
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "qv": ["q_proj", "v_proj"],
}


def is_quantized(method: str) -> bool:
    return method in ("qlora", "qdora")


def build_lora_config(method: str, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                      targets: str = "all-linear"):
    from peft import LoraConfig, TaskType

    target_modules = TARGET_PRESETS.get(targets, targets.split(","))
    kw = dict(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules,
              bias="none", task_type=TaskType.CAUSAL_LM)
    if method in ("dora", "qdora"):
        kw["use_dora"] = True
    elif method == "rslora":
        kw["use_rslora"] = True
    elif method == "pissa":
        kw["init_lora_weights"] = "pissa_niter_4"   # fast randomized SVD init
        kw["lora_dropout"] = 0.0                     # PiSSA recommends no dropout
    elif method == "olora":
        kw["init_lora_weights"] = "olora"
    elif method not in ("lora", "qlora", "loraplus"):
        raise ValueError(f"unknown method {method}; choose from {list(LORA_METHODS)}")
    return LoraConfig(**kw)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_tokenizer(path: str = DEFAULT_BASE):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(base: str = DEFAULT_BASE, adapter: str | None = None, quantize_4bit: bool = False,
               attn: str = "sdpa", for_training: bool = False, merge: bool = False):
    """Load the base model, optionally 4-bit, optionally with a trained LoRA adapter on top."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kw = dict(dtype=torch.bfloat16, attn_implementation=attn)
    if quantize_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if not quantize_4bit and torch.cuda.is_available():
        model.to("cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=for_training)
        if merge and not quantize_4bit:
            model = model.merge_and_unload()
    if not for_training:
        model.eval()
    return model


def resolve_adapter(path: str | None) -> tuple[str, str | None]:
    """Accept a base-model dir, an adapter dir (adapter_config.json), or a run dir (uses run/adapter).
    Returns (base_model_path, adapter_path_or_None)."""
    if not path:
        return DEFAULT_BASE, None
    for cand in (path, os.path.join(path, "adapter"), os.path.join(path, "final")):
        cfg = os.path.join(cand, "adapter_config.json")
        if os.path.exists(cfg):
            with open(cfg) as f:
                base = json.load(f).get("base_model_name_or_path") or DEFAULT_BASE
            if not os.path.isdir(base):
                base = DEFAULT_BASE
            return base, cand
    return path, None  # a plain (merged / base) model dir


# --------------------------------------------------------------------------- #
# Chat encoding with assistant-only labels
# --------------------------------------------------------------------------- #
_ASSIST_SPAN = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)


def render(tok, messages: list[dict], add_generation_prompt: bool = False, enable_thinking: bool = True) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   enable_thinking=enable_thinking)


def encode_chat(tok, messages: list[dict], max_len: int) -> dict | None:
    """Render with the Qwen3 template and supervise only assistant turns (incl. <|im_end|>).
    The template keeps <think> only for the final assistant turn and inserts an empty
    '<think>\\n\\n</think>' when the final answer has no reasoning (= Qwen3 non-thinking format)."""
    text = render(tok, messages)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=max_len)
    spans = [(m.start(1), m.end(1)) for m in _ASSIST_SPAN.finditer(text)]
    if not spans:
        return None
    labels = [IGNORE] * len(enc["input_ids"])
    n_sup = 0
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if any(a <= s < b for a, b in spans):
            labels[i] = enc["input_ids"][i]
            n_sup += 1
    if n_sup == 0:  # assistant answer was truncated away entirely
        return None
    return {"input_ids": enc["input_ids"], "labels": labels}


@dataclass
class ChatCollator:
    pad_id: int

    def __call__(self, batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), IGNORE, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
_THINK = re.compile(r"<think>(.*?)</think>\s*", re.S)


def split_thinking(text: str) -> tuple[str, str]:
    m = _THINK.search(text)
    if not m:
        if "<think>" in text:  # unfinished thinking (hit max_new_tokens)
            return text.split("<think>", 1)[1].strip(), ""
        return "", text.strip()
    return m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()


@torch.inference_mode()
def generate(model, tok, conversations: list[list[dict]], max_new_tokens: int = 1024,
             enable_thinking: bool = False, temperature: float = 0.0, top_p: float = 0.95,
             batch_size: int = 8, repetition_penalty: float = 1.0) -> list[str]:
    """Batched generation (left padding). temperature=0 -> greedy."""
    tok.padding_side = "left"
    prompts = [render(tok, c, add_generation_prompt=True, enable_thinking=enable_thinking) for c in conversations]
    outs: list[str] = [""] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gk = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, repetition_penalty=repetition_penalty)
        if temperature > 0:
            gk.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=20)
        else:
            gk.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        out = model.generate(**enc, **gk)
        for j, i in enumerate(idx):
            outs[i] = tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = "right"
    return outs


# --------------------------------------------------------------------------- #
def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def level_system_prompt(level: str | None) -> str | None:
    if not level:
        return None
    import sys
    sys.path.insert(0, ROOT)
    from ai_qa_data.taxonomy import LEVELS
    return LEVELS[level]["system"] if level in LEVELS else None
