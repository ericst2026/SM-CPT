"""Continued pre-training (CPT) of a Qwen3 base model with a high-rank LoRA adapter on the collected
document corpora (corpus/ai + corpus/replay), with an MMLU evaluation after every epoch.

    python -m cpt_trainer.train --help
"""
