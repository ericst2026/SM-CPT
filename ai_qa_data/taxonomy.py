"""Label spaces used across the pipeline: AI topics, learner levels, buckets.

Everything that is "a list of labels + a natural-language description" lives here
so that the keyword stage, the zero-shot stage and the docs stay consistent.
"""

# --------------------------------------------------------------------------- #
# Output buckets (what the record will be used for when training)
# --------------------------------------------------------------------------- #
BUCKETS = {
    "ai_sft":         "AI question/answer + explanation data for supervised fine-tuning",
    "ai_rl_math":     "AI-relevant math problems with a verifiable final answer (RL / GRPO)",
    "ai_rl_code":     "AI-relevant Python problems with unit tests (RL / GRPO)",
    "ai_eval":        "Held-out benchmark items (never train on these)",
    "general_replay": "Small sample of non-AI data kept to prevent catastrophic forgetting",
}

# --------------------------------------------------------------------------- #
# AI relevance domains: which "kind" of AI relevance a record has.
# --------------------------------------------------------------------------- #
DOMAIN_HYPOTHESES = {
    "ai_core": "This text is about artificial intelligence, machine learning, deep learning, or data science.",
    "ai_math": "This is a math problem about linear algebra, probability, statistics, calculus, or optimization of the kind used in machine learning.",
    "ai_code": "This is a programming task about machine learning, neural networks, or data analysis in Python.",
}

# --------------------------------------------------------------------------- #
# Topic (content type) labels -- multi-label.
# --------------------------------------------------------------------------- #
TOPICS = {
    "concepts":       "This text explains what an artificial intelligence or machine learning concept means.",
    "applications":   "This text describes a real-world application or use case of artificial intelligence.",
    "theory_math":    "This text contains mathematical theory, formulas, or a derivation used in machine learning.",
    "history":        "This text is about the history of artificial intelligence, its pioneers, or milestones.",
    "coding":         "This text contains programming code for machine learning or data science.",
    "ethics_society": "This text discusses ethics, safety, bias, law, jobs, or the societal impact of artificial intelligence.",
    "research":       "This text discusses AI research papers, experiments, benchmarks, or state-of-the-art results.",
}

# --------------------------------------------------------------------------- #
# Learner levels (the 8 audiences the model has to serve).
# `zs` = zero-shot hypothesis, `fk` = target Flesch-Kincaid grade range,
# `system` = system prompt used when the final SFT mix is built.
# --------------------------------------------------------------------------- #
LEVELS = {
    "primary_school": {
        "zs": "This text is written in very simple words for young children in primary school.",
        "fk": (0, 5.5),
        "system": "You are a friendly AI tutor for primary school children (ages 6-11). Use very simple words, short sentences, everyday examples and no math formulas.",
    },
    "middle_school": {
        "zs": "This text is written for middle school students aged 11 to 14.",
        "fk": (5.5, 8.5),
        "system": "You are an AI tutor for middle school students (ages 11-14). Explain clearly with relatable examples and only basic math.",
    },
    "high_school": {
        "zs": "This text is written for high school students.",
        "fk": (8.5, 12.5),
        "system": "You are an AI tutor for high school students. Use precise terms, simple algebra, and small Python examples when helpful.",
    },
    "undergrad_non_ai": {
        "zs": "This text is written for university students who are not specialists in computer science.",
        "fk": (12.5, 15.0),
        "system": "You are an AI tutor for university students from non-AI majors. Explain the intuition first, then the key ideas, with light math.",
    },
    "ai_student": {
        "zs": "This text is written for computer science or AI students learning machine learning in depth.",
        "fk": (12.5, 18.0),
        "system": "You are an AI teaching assistant for computer science / AI students. Be rigorous: give definitions, derivations and Python code where useful.",
    },
    "general_public": {
        "zs": "This text is written for ordinary adults with no technical background.",
        "fk": (6.0, 12.0),
        "system": "You are an AI explainer for the general public. Avoid jargon, focus on what AI does in daily life, its benefits and its risks.",
    },
    "researcher_non_ai": {
        "zs": "This text is written for scientists from other fields who want to apply AI in their own research.",
        "fk": (13.0, 20.0),
        "system": "You are an AI research consultant for scientists from other fields. Be precise and practical about methods, data, evaluation and pitfalls.",
    },
    "ai_researcher": {
        "zs": "This text is written for expert AI researchers and assumes advanced technical knowledge.",
        "fk": (15.0, 30.0),
        "system": "You are an expert AI research assistant. Be technically rigorous, use formal notation, reference known methods and discuss trade-offs.",
    },
}

LEVEL_NAMES = list(LEVELS)

# Coarse school grade -> level (for sources that carry a grade field)
def grade_to_level(grade: int) -> str:
    if grade <= 5:
        return "primary_school"
    if grade <= 8:
        return "middle_school"
    if grade <= 12:
        return "high_school"
    return "undergrad_non_ai"
