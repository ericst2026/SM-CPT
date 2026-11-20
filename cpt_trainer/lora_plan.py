#!/usr/bin/env python3
"""Per-layer / per-module LoRA rank plans (YAML) -> PEFT LoraConfig with rank_pattern / alpha_pattern.

    python -m cpt_trainer.lora_plan configs/lora/cpt_qwen06b_rs_adaptive_v1.yaml            # rank table + params
    python -m cpt_trainer.lora_plan configs/lora/cpt_qwen06b_rs_adaptive_v1.yaml --base models/Qwen3-1.7B-Base

The file format is documented at the top of configs/lora/cpt_qwen06b_rs_adaptive_v1.yaml. train.py uses
it through --lora-config; every (module, layer) gets an explicit rank_pattern entry, and after
get_peft_model the ranks found in the model are checked against the plan.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_qa_train.common import ALL_LINEAR, LORA_METHODS, TARGET_PRESETS, ROOT  # noqa: E402

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
BLOCK = {m: ("self_attn" if m in ATTN else "mlp") for m in ALL_LINEAR}
KEYS = {"name", "method", "alpha", "dropout", "default_r", "rank_bounds", "targets", "ranks", "alpha_overrides"}


def parse_layers(spec, num_layers: int) -> list[int]:
    """"all" | "a-b" | "a-b,c,d-e" | int -> sorted layer indices."""
    if isinstance(spec, int):
        parts = [str(spec)]
    elif str(spec).strip().lower() == "all":
        return list(range(num_layers))
    else:
        parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    layers: set[int] = set()
    for p in parts:
        lo, _, hi = p.partition("-")
        try:
            a, b = int(lo), int(hi or lo)
        except ValueError:
            raise ValueError(f"bad layer spec {spec!r}") from None
        if not 0 <= a <= b < num_layers:
            raise ValueError(f"layer range {p!r} outside 0-{num_layers - 1}")
        layers.update(range(a, b + 1))
    return sorted(layers)


def _expand(section: dict, field: str, targets: list[str], num_layers: int, default) -> dict:
    table = {}
    for mod, rules in (section or {}).items():
        if mod not in targets:
            raise ValueError(f"`{mod}` has {field} rules but is not in targets {targets}")
        if not isinstance(rules, list):
            raise ValueError(f"{mod}: expected a list of {{layers, {field}}} entries")
        for rule in rules:
            if set(rule) != {"layers", field}:
                raise ValueError(f"{mod}: each entry needs exactly `layers` and `{field}`, got {sorted(rule)}")
            for layer in parse_layers(rule["layers"], num_layers):
                if (mod, layer) in table:
                    raise ValueError(f"{mod} layer {layer} is listed twice")
                table[(mod, layer)] = rule[field]
    return {(m, l): table.get((m, l), default) for m in targets for l in range(num_layers)}


def load_plan(path: str, num_layers: int) -> dict:
    """Read + validate a plan; returns the plan with `rank_table` / `alpha_table` {(module, layer): value}."""
    import yaml

    with open(path) as f:
        plan = yaml.safe_load(f) or {}
    unknown = set(plan) - KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)} (allowed: {sorted(KEYS)})")
    if "default_r" not in plan:
        raise ValueError(f"{path}: `default_r` is required")
    method = plan.get("method")
    if method is not None and method not in LORA_METHODS:
        raise ValueError(f"{path}: unknown method {method!r}; choose from {list(LORA_METHODS)}")
    targets = plan.get("targets", ALL_LINEAR)
    targets = TARGET_PRESETS.get(targets, targets) if isinstance(targets, str) else list(targets)
    bad = [t for t in targets if t not in ALL_LINEAR]
    if bad:
        raise ValueError(f"{path}: unknown target modules {bad} (Qwen3 linear layers: {ALL_LINEAR})")
    plan["targets"] = targets
    plan["path"] = os.path.abspath(path)
    plan["num_layers"] = num_layers
    plan["rank_table"] = _expand(plan.get("ranks"), "r", targets, num_layers, plan["default_r"])
    plan["alpha_table"] = _expand(plan.get("alpha_overrides"), "alpha", targets, num_layers, None)
    lo, hi = plan.get("rank_bounds", [1, 10**9])
    for (m, l), r in plan["rank_table"].items():
        if not isinstance(r, int) or not lo <= r <= hi:
            raise ValueError(f"{path}: {m} layer {l} has r={r!r}, outside rank_bounds [{lo}, {hi}]")
    return plan


def module_name(mod: str, layer: int) -> str:
    return f"model.layers.{layer}.{BLOCK[mod]}.{mod}"


def build_config(plan: dict, method: str, alpha: int, dropout: float):
    """LoraConfig with an explicit rank_pattern entry per adapted module (and alpha_pattern if given)."""
    from ai_qa_train.common import build_lora_config

    cfg = build_lora_config(method, plan["default_r"], alpha, dropout, ",".join(plan["targets"]))
    # PEFT matches pattern keys as regex `(.*\.)?(key)$`, so the full dotted names are unambiguous
    # ("layers.1.mlp" cannot match "layers.11.mlp").
    cfg.rank_pattern = {module_name(m, l): r for (m, l), r in plan["rank_table"].items()}
    cfg.alpha_pattern = {module_name(m, l): a for (m, l), a in plan["alpha_table"].items() if a is not None}
    return cfg


def model_shapes(base: str) -> tuple[int, dict]:
    """num_hidden_layers and (in, out) features of each linear module, from the base config.json."""
    with open(os.path.join(base, "config.json")) as f:
        c = json.load(f)
    h, inter = c["hidden_size"], c["intermediate_size"]
    hd = c.get("head_dim") or h // c["num_attention_heads"]
    q, kv = c["num_attention_heads"] * hd, c["num_key_value_heads"] * hd
    shapes = {"q_proj": (h, q), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (q, h),
              "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h)}
    return c["num_hidden_layers"], shapes


def param_count(rank_table: dict, shapes: dict) -> int:
    return sum(r * sum(shapes[m]) for (m, _), r in rank_table.items())


def ranks_in_model(model) -> dict:
    """{(module, layer): r} read back from the lora_A weights of a PEFT model."""
    found = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            parts = name.split(".")
            if "layers" in parts:
                layer = int(parts[parts.index("layers") + 1])
                found[(parts[-1], layer)] = mod.lora_A["default"].weight.shape[0]
    return found


def check_model(model, plan: dict) -> None:
    found, want = ranks_in_model(model), plan["rank_table"]
    if found != want:
        diff = sorted(k for k in set(found) | set(want) if found.get(k) != want.get(k))[:8]
        raise RuntimeError("LoRA ranks in the model differ from the plan: "
                           + ", ".join(f"{m} L{l}: model {found.get((m, l))} vs plan {want.get((m, l))}" for m, l in diff))


def scale_of(method: str, alpha: float, r: int) -> float:
    return alpha / r ** 0.5 if method == "rslora" else alpha / r


def format_table(plan: dict, shapes: dict, method: str, alpha: float) -> str:
    """Layer x module rank grid, parameter totals and the resulting scale per rank."""
    mods, n = plan["targets"], plan["num_layers"]
    w = max(9, *(len(m) + 1 for m in mods))
    lines = ["layer " + "".join(m.rjust(w) for m in mods) + "   M params"]
    for l in range(n):
        row = [plan["rank_table"][(m, l)] for m in mods]
        p = sum(r * sum(shapes[m]) for m, r in zip(mods, row)) / 1e6
        lines.append(f"{l:5d} " + "".join(str(r).rjust(w) for r in row) + f"   {p:8.2f}")
    per_mod = [sum(plan["rank_table"][(m, l)] * sum(shapes[m]) for l in range(n)) / 1e6 for m in mods]
    lines.append("M par " + "".join(f"{p:.2f}".rjust(w) for p in per_mod))
    total = param_count(plan["rank_table"], shapes)
    uni = {r: r * sum(sum(shapes[m]) for m in mods) * n for r in (64, 128, 256)}
    ranks = sorted(set(plan["rank_table"].values()))
    lines += ["", f"total {total / 1e6:.1f} M trainable params ({total * 2 / 2**20:.0f} MiB as bf16 adapter); "
              + ", ".join(f"uniform r={r}: {v / 1e6:.1f} M" for r, v in uni.items()),
              f"{method} alpha={alpha}: scale " + ", ".join(f"r{r}={scale_of(method, alpha, r):.3g}" for r in ranks)]
    if any(a is not None for a in plan["alpha_table"].values()):
        lines.append("alpha overrides: " + ", ".join(f"{m} L{l}={a}" for (m, l), a in plan["alpha_table"].items() if a is not None))
    return "\n".join(lines)


def resolved_json(plan: dict, method: str, alpha: float, dropout: float, shapes: dict) -> dict:
    """What actually goes into the run (saved as <run>/lora_plan.json)."""
    return {"name": plan.get("name"), "source": os.path.relpath(plan["path"], ROOT), "method": method,
            "alpha": alpha, "dropout": dropout, "default_r": plan["default_r"], "targets": plan["targets"],
            "trainable_params": param_count(plan["rank_table"], shapes),
            "ranks": {m: [plan["rank_table"][(m, l)] for l in range(plan["num_layers"])] for m in plan["targets"]},
            "alpha_overrides": {module_name(m, l): a for (m, l), a in plan["alpha_table"].items() if a is not None}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", help="YAML rank plan")
    ap.add_argument("--base", default=os.path.join(ROOT, "models", "Qwen3-0.6B-Base"), help="base model dir (for shapes)")
    args = ap.parse_args()
    n, shapes = model_shapes(args.base)
    plan = load_plan(args.plan, n)
    method, alpha = plan.get("method", "lora"), plan.get("alpha", 16)
    print(f"{plan.get('name', os.path.basename(args.plan))}: {method}, alpha {alpha}, dropout {plan.get('dropout', 0.05)}, "
          f"{len(plan['targets'])} targets x {n} layers\n")
    print(format_table(plan, shapes, method, alpha))


if __name__ == "__main__":
    main()
