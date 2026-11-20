"""Catalog of source datasets + per-source normalizers and metadata rules.

Every source is described by a `Source`:
  * where it lives on the Hugging Face Hub (hf_id / config / split)
  * how one raw row maps to the unified record   -> `normalize(row) -> dict | list[dict] | None`
  * a metadata rule using the source's own fields -> `meta_rule(row) -> "accept" | "reject" | None`
      accept = AI-relevant by construction (e.g. ai.stackexchange, subject == machine_learning)
      reject = certainly out of scope (e.g. role-play, jailbreak data, non-English)
      None   = undecided -> keyword + zero-shot stages decide
  * a learner-level prior derived from metadata   -> `level_prior(row) -> dict[level, weight]`
  * which output bucket it feeds (ai_sft / ai_rl_math / ai_rl_code / ai_eval)
  * `pool`: "curated" (small, AI-specific, keep everything that passes) or
            "huge" (general corpus, streamed and keyword-prefiltered while downloading)

Unified record (written by collect_datasets.py):
{
  "id": "<source>/<n>", "source": str, "hf": "id:config:split", "bucket": str, "domain": str,
  "task": "qa|chat|mcq|explain|code|rl_math|rl_code",
  "lang": "en", "question": str, "answer": str, "reasoning": str|None,
  "ground_truth": str|None, "tests": str|None, "messages": list|None,
  "meta": {...source specific...},
  "hints": {"meta_decision": "accept|reject|None", "level_prior": {...}}
}
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

Row = dict[str, Any]

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.S)
_BOXED_RE = re.compile(r"\\boxed\{")


def split_think(text: str) -> tuple[str | None, str]:
    """'<think>r</think> a' -> ('r', 'a'). Also handles R1 dumps that only contain the closing tag."""
    if not text:
        return None, ""
    m = _THINK_RE.search(text)
    if not m:
        if "</think>" in text:
            r, a = text.split("</think>", 1)
            return (r.replace("<think>", "").strip() or None), a.strip()
        return None, text.strip()
    reasoning = m.group(1).strip() or None
    return reasoning, (text[:m.start()] + text[m.end():]).strip()


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (handles nested braces)."""
    if not text:
        return None
    starts = [m.end() for m in _BOXED_RE.finditer(text)]
    if not starts:
        return None
    i = starts[-1]
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1].strip() if depth == 0 else None


def as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                v = loader(x)
                if isinstance(v, list):
                    return v
            except Exception:
                pass
    return []


def chat_to_qa(messages: list[dict], role_key="role", content_key="content",
               user=("user", "human"), assistant=("assistant", "gpt", "model")) -> dict | None:
    """Multi-turn chat -> first user turn as question, first assistant reply as answer,
    full conversation kept in `messages` (normalised to role/content)."""
    norm = []
    for m in messages or []:
        r = str(m.get(role_key, "")).lower()
        c = m.get(content_key) or ""
        if r in user:
            norm.append({"role": "user", "content": c})
        elif r in assistant:
            norm.append({"role": "assistant", "content": c})
        elif r == "system":
            norm.append({"role": "system", "content": c})
    q = next((m["content"] for m in norm if m["role"] == "user"), None)
    a = next((m["content"] for m in norm if m["role"] == "assistant"), None)
    if not q or not a:
        return None
    reasoning, a = split_think(a)
    turns = sum(1 for m in norm if m["role"] == "user")
    return {"question": q, "answer": a, "reasoning": reasoning,
            "messages": norm if turns > 1 else None, "task": "chat"}


def mcq(question: str, choices: list[str], answer_idx: int | None) -> dict:
    letters = "ABCDEFGHIJ"
    body = question.strip() + "\n\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    ans = None
    if answer_idx is not None and 0 <= answer_idx < len(choices):
        ans = f"{letters[answer_idx]}. {choices[answer_idx]}"
    return {"question": body, "answer": ans, "ground_truth": letters[answer_idx] if ans else None, "task": "mcq"}


# --------------------------------------------------------------------------- #
# Source spec
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    name: str
    hf_id: str
    normalize: Callable[[Row], dict | list[dict] | None]
    config: str | None = None
    split: str = "train"
    bucket: str = "ai_sft"            # ai_sft | ai_rl_math | ai_rl_code | ai_eval
    domain: str = "ai_core"           # ai_core | ai_math | ai_code | reasoning  (which relevance test applies)
    pool: str = "curated"             # curated | huge
    meta_rule: Callable[[Row], str | None] = lambda r: None
    level_prior: Callable[[Row], dict | None] = lambda r: None
    max_rows: int | None = None       # cap on rows *written*
    max_scan: int | None = None       # cap on rows *read* (huge pools)
    columns: list[str] | None = None  # read only these columns (faster streaming)
    lang: str = "en"
    enabled: bool = True
    desc: str = ""
    license: str = ""
    added_in: str = "v0.1"            # data version that introduced the source
    # keep rows that fail the AI test but are good general reasoning (ai_qa_data/reasoning.py) for the
    # reasoning_sft bucket; always on for domain="reasoning" sources
    reasoning: bool = False

    def __post_init__(self):
        if self.domain == "reasoning":
            self.reasoning = True

    @property
    def hf(self) -> str:
        return f"{self.hf_id}:{self.config or 'default'}:{self.split}"


ACCEPT = lambda r: "accept"  # noqa: E731
P = lambda **kw: (lambda r: kw)  # noqa: E731  constant level prior


# --------------------------------------------------------------------------- #
# Normalizers / rules for specific sources
# --------------------------------------------------------------------------- #
def n_qa(qk, ak, task="qa"):
    def f(r):
        q, a = (r.get(qk) or "").strip(), (r.get(ak) or "").strip()
        return {"question": q, "answer": a, "task": task} if q and a else None
    return f


def n_stackexchange(r):
    tb = (r.get("title_body") or "").strip()
    a = (r.get("upvoted_answer") or "").strip()
    return {"question": tb, "answer": a, "task": "qa"} if tb and a else None


def n_razin(r):
    q = f"{r.get('Title', '').strip()}\n\n{(r.get('Question') or '').strip()}".strip()
    a = (r.get("Answer") or "").strip()
    return {"question": q, "answer": a, "task": "qa",
            "meta": {"tags": r.get("Tags"), "score": r.get("Score"), "is_accepted": r.get("is_accepted")}} if q and a else None


def n_gemma_text(r):
    t = r.get("text") or ""
    m = re.search(r"<start_of_turn>user\s*(.*?)<end_of_turn>\s*<start_of_turn>model\s*(.*?)(?:<end_of_turn>|$)", t, re.S)
    if not m:
        return None
    return {"question": m.group(1).strip(), "answer": m.group(2).strip(), "task": "qa"}


def n_inst_text(r):
    t = r.get("text") or ""
    m = re.search(r"\[INST\](.*?)\[/INST\](.*?)(?:</s>|$)", t, re.S)
    if not m:
        return None
    return {"question": m.group(1).strip(), "answer": m.group(2).strip(), "task": "qa"}


def n_ai_history_timeline(r):
    title, year = r.get("title") or r.get("name"), r.get("year")
    summary, impact = (r.get("summary") or "").strip(), (r.get("impact") or "").strip()
    if not title or not summary:
        return None
    who = f" by {r['organization']}" if r.get("organization") else ""
    q = f"What was \"{title}\"{who} ({year}), and why does it matter in the history of AI?"
    a = summary + (f"\n\nWhy it matters: {impact}" if impact else "")
    return {"question": q, "answer": a, "task": "qa",
            "meta": {"year": year, "category": r.get("category"), "significance": r.get("significance_score")}}


def n_arxiv_abstract(r):
    t, a = (r.get("title") or "").strip(), (r.get("abstract") or "").strip()
    if not t or len(a) < 200:
        return None
    t = re.sub(r"\s+", " ", t)
    return {"question": f"What is the research paper \"{t}\" about, and what does it contribute?",
            "answer": re.sub(r"\s+", " ", a), "task": "qa"}


_DL_CODE_RE = re.compile(r"\b(?:torch|nn\.|tensorflow|tf\.|keras|jax|flax|optax|sklearn|numpy|np\.)")


def n_arxivdl(r):
    p, code = (r.get("prompt") or "").strip(), (r.get("function") or "").strip()
    if not p or not code or len(code) > 6000:
        return None
    desc = (r.get("description") or "").strip()
    return {"question": p, "answer": f"```python\n{code}\n```\n\n{desc}".strip(), "task": "code",
            "meta": {"repo": r.get("repo"), "file": r.get("file")}}


def rule_arxivdl(r):
    return "accept" if _DL_CODE_RE.search(r.get("function") or "") else None


MMLU_AI = {"machine_learning"}
MMLU_MAYBE = {"high_school_computer_science", "college_computer_science", "high_school_statistics",
              "college_mathematics", "econometrics", "computer_security", "electrical_engineering"}


def n_mmlu(r):
    d = mcq(r["question"], list(r["choices"]), int(r["answer"]))
    d["meta"] = {"subject": r.get("subject")}
    return d


def rule_mmlu(r):
    s = r.get("subject")
    return "accept" if s in MMLU_AI else (None if s in MMLU_MAYBE else "reject")


def prior_mmlu(r):
    s = r.get("subject") or ""
    if s.startswith("high_school"):
        return {"high_school": 1.0}
    if s == "machine_learning":
        return {"ai_student": 1.0}
    return {"undergrad_non_ai": 1.0}


def n_mmlu_pro(r):
    opts = list(r.get("options") or [])
    d = mcq(r["question"], opts, r.get("answer_index"))
    d["reasoning"] = r.get("cot_content") or None
    d["meta"] = {"category": r.get("category"), "src": r.get("src")}
    return d


def rule_mmlu_pro(r):
    return None if r.get("category") in ("computer science", "math", "engineering") else "reject"


def n_ds1000(r):
    md = r.get("metadata") or {}
    if isinstance(md, str):
        md = ast.literal_eval(md)
    return {"question": r["prompt"], "answer": r.get("reference_code"), "tests": r.get("code_context"),
            "task": "code", "meta": {"library": md.get("library"), "perturbation": md.get("perturbation_type")}}


# --- Cosmopedia: textbook units with an explicit `audience` field --------------
_COSMO_TITLE = re.compile(r'textbook on\s+"(.+?)"', re.S)
_COSMO_SUB = re.compile(r'(?:current sub-unit|writing the (?:first|current) chapter|sub-unit)[^"]*"([^"]+)"', re.S)
_AUD_PHRASE = {
    "young_children": "young children", "children": "children", "middle_school_students": "middle school students",
    "high_school_studnets": "high school students", "high_school_students": "high school students",
    "college_students": "college students", "college_studnets": "college students",
    "researchers": "researchers", "general": "a general audience",
}


def n_cosmopedia(r):
    prompt, text = (r.get("prompt") or "").strip(), (r.get("text") or "").strip()
    if not prompt or len(text) < 300:
        return None
    aud = r.get("audience") or "general"
    title_m, sub_m = _COSMO_TITLE.search(prompt), _COSMO_SUB.search(prompt)
    if title_m:  # turn "write a textbook unit" into a direct teaching question
        topic = sub_m.group(1).strip() if sub_m else title_m.group(1).strip()
        course = title_m.group(1).strip()
        q = f"Explain \"{topic}\"" + (f" (from the course \"{course}\")" if sub_m else "") + \
            f" for {_AUD_PHRASE.get(aud, 'a general audience')}."
    else:
        q = prompt
    return {"question": q, "answer": text, "task": "explain",
            "meta": {"audience": aud, "format": r.get("format"), "seed_data": r.get("seed_data"),
                     "orig_prompt": prompt[:2000]}}


def prior_cosmopedia(r):
    from .levels import COSMOPEDIA_AUDIENCE
    return COSMOPEDIA_AUDIENCE.get(r.get("audience") or "", None)


# --- Chat corpora ------------------------------------------------------------
def n_openhermes(r):
    d = chat_to_qa(as_list(r.get("conversations")), role_key="from", content_key="value")
    if d:
        d["meta"] = {"category": r.get("category"), "orig_source": r.get("source"), "topic": r.get("topic")}
    return d


OPENHERMES_REJECT = {"roleplay", "rp", "joke", "song", "wordgame", "riddle", "greeting", "card", "gtkm",
                     "awareness", "experience", "stylized_response", "writing", "editor", "theory_of_mind",
                     "counterfactual_contextual", "trivia"}


def rule_openhermes(r):
    return "reject" if (r.get("category") or "") in OPENHERMES_REJECT else None


def n_messages(r, key="messages"):
    d = chat_to_qa(as_list(r.get(key)))
    if d:
        d["meta"] = {"orig_source": r.get("source")}
    return d


TULU_REJECT = ("wildjailbreak", "wildguard", "aya", "hard_coded", "coconot", "table_gpt", "ifdata", "personahub_ifdata")


def rule_tulu(r):
    s = r.get("source") or ""
    return "reject" if any(x in s for x in TULU_REJECT) else None


SMOLTALK_REJECT = {"smol-rewrite", "smol-summarize", "smol-constraints", "apigen-80k", "longalign",
                   "explore-instruct-rewriting", "everyday-conversations"}


def rule_smoltalk(r):
    return "reject" if (r.get("source") or "") in SMOLTALK_REJECT else None


def n_wildchat(r):
    d = chat_to_qa(as_list(r.get("conversation")))
    if d:
        d["meta"] = {"model": r.get("model"), "country": r.get("country")}
    return d


def rule_wildchat(r):
    if r.get("language") != "English" or r.get("toxic") or r.get("redacted"):
        return "reject"
    return None


def n_nemotron(r):
    msgs = as_list(r.get("input"))
    q = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
    reasoning, a = split_think(r.get("output") or "")
    if not q or not a:
        return None
    gt = last_boxed(a) if r.get("category") == "math" else mcq_letter_from_answer(q, a)
    return {"question": q, "answer": a, "reasoning": reasoning, "ground_truth": gt, "task": "chat",
            "meta": {"category": r.get("category"), "reasoning_mode": r.get("reasoning"), "generator": r.get("generator")}}


def n_openthoughts(r):
    q = r.get("problem")
    a = r.get("deepseek_solution") or ""
    if not q or not a:
        return None
    dom = r.get("domain")
    return {"question": q, "answer": a, "reasoning": r.get("deepseek_reasoning") or None,
            "ground_truth": (r.get("ground_truth_solution") if dom == "math" else None),
            "tests": (r.get("test_cases") if dom == "code" else None), "task": "chat",
            "meta": {"domain": dom, "orig_source": r.get("source")}}


def n_dolly(r):
    q = (r.get("instruction") or "").strip()
    if r.get("context"):
        q = f"{q}\n\nContext:\n{r['context'].strip()}"
    a = (r.get("response") or "").strip()
    return {"question": q, "answer": a, "task": "qa", "meta": {"category": r.get("category")}} if q and a else None


def n_instr_io(r, ik="instruction", ink="input", ok="output", task="code"):
    q = (r.get(ik) or "").strip()
    if r.get(ink):
        q = f"{q}\n\n{str(r[ink]).strip()}"
    a = (r.get(ok) or "").strip()
    return {"question": q, "answer": a, "task": task} if q and a else None


def n_webinstruct(r):
    q, a = (r.get("question") or "").strip(), (r.get("answer") or "").strip()
    return {"question": q, "answer": a, "task": "qa", "meta": {"orig_source": r.get("source")}} if q and a else None


def prior_webinstruct(r):
    return {"high_school": 1.0} if r.get("source") == "socratic" else {"undergrad_non_ai": 0.5, "ai_student": 0.5}


# --- RL: code -----------------------------------------------------------------
def n_kodcode(r):
    q, sol, test = r.get("question"), r.get("solution"), r.get("test")
    if not q or not test:
        return None
    return {"question": q, "answer": f"```python\n{sol}\n```" if sol else None, "tests": test, "task": "rl_code",
            "meta": {"subset": r.get("subset"), "difficulty": r.get("gpt_difficulty"),
                     "pass_rate": r.get("gpt_pass_percentage"), "test_info": r.get("test_info"),
                     "question_id": r.get("question_id")}}


# --- RL: math -----------------------------------------------------------------
AI_MATH_TOPICS = ("Linear Algebra", "Probability", "Statistics", "Differential Calculus", "Multivariable",
                  "Integral Calculus", "Optimization", "Information", "Matrices", "Vectors", "Applied Mathematics")
NON_AI_MATH_TOPICS = ("Geometry", "Number Theory", "Abstract Algebra", "Group Theory", "Field Theory",
                      "Ring Theory", "Topology", "Logic", "Combinatorics -> Other")


def n_deepmath(r):
    if not r.get("question") or not r.get("final_answer"):
        return None
    return {"question": r["question"], "answer": None, "reasoning": r.get("r1_solution_1"),
            "ground_truth": str(r["final_answer"]), "task": "rl_math",
            "meta": {"topic": r.get("topic"), "difficulty": r.get("difficulty")}}


def rule_deepmath(r):
    t = r.get("topic") or ""
    if any(x in t for x in AI_MATH_TOPICS):
        return "accept"
    if any(x in t for x in NON_AI_MATH_TOPICS):
        return "reject"
    return None


def prior_math_difficulty(r):
    try:
        d = float(r.get("difficulty") or 5)
    except ValueError:
        d = 5
    if d < 3:
        return {"high_school": 1.0}
    if d < 6:
        return {"ai_student": 0.6, "undergrad_non_ai": 0.4}
    return {"ai_researcher": 0.6, "ai_student": 0.4}


def n_numina(r):
    sol = r.get("solution") or ""
    gt = last_boxed(sol)
    if not r.get("problem") or not gt:
        return None
    return {"question": r["problem"], "answer": sol, "ground_truth": gt, "task": "rl_math",
            "meta": {"orig_source": r.get("source")}}


def prior_numina(r):
    s = r.get("source") or ""
    if s in ("gsm8k", "orca_math"):
        return {"middle_school": 1.0}
    if s in ("cn_k12", "synthetic_math", "math"):
        return {"high_school": 1.0}
    return {"high_school": 0.5, "ai_student": 0.5}


def n_openr1(r):
    if not r.get("problem") or not r.get("answer"):
        return None
    return {"question": r["problem"], "answer": r.get("solution"), "ground_truth": str(r["answer"]), "task": "rl_math",
            "meta": {"problem_type": r.get("problem_type"), "question_type": r.get("question_type"),
                     "orig_source": r.get("source")}}


def rule_openr1(r):
    return "reject" if r.get("problem_type") in ("Geometry", "Number Theory", "Logic and Puzzles") else None


def n_deepscaler(r):
    if not r.get("problem") or not r.get("answer"):
        return None
    return {"question": r["problem"], "answer": r.get("solution"), "ground_truth": str(r["answer"]), "task": "rl_math"}


# --------------------------------------------------------------------------- #
# THE CATALOG
# --------------------------------------------------------------------------- #
SE = "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl"

SOURCES: list[Source] = [
    # ======================= A. Curated AI Q&A (pool=curated) =======================
    Source("se_ai", SE, n_stackexchange, config="ai", meta_rule=ACCEPT,
           level_prior=P(ai_student=0.7, general_public=0.3),
           desc="ai.stackexchange.com best-voted answers: concepts, theory, philosophy, history of AI"),
    Source("se_datascience", SE, n_stackexchange, config="datascience", meta_rule=ACCEPT,
           level_prior=P(ai_student=0.7, researcher_non_ai=0.3),
           desc="datascience.stackexchange.com: practical ML / DL / data science"),
    Source("se_stats", SE, n_stackexchange, config="stats",
           level_prior=P(researcher_non_ai=0.5, ai_student=0.5),
           desc="stats.stackexchange (CrossValidated): ML theory & derivations (filtered: also has pure stats)"),
    Source("se_cs", SE, n_stackexchange, config="cs", level_prior=P(ai_student=1.0),
           desc="cs.stackexchange: algorithms/theory; AI part selected by filter"),
    Source("se_cstheory", SE, n_stackexchange, config="cstheory", level_prior=P(ai_researcher=1.0),
           desc="cstheory.stackexchange: learning theory part selected by filter"),
    Source("se_robotics", SE, n_stackexchange, config="robotics", level_prior=P(ai_student=1.0),
           desc="robotics.stackexchange: control, SLAM, RL for robots (filtered)"),
    Source("se_philosophy", SE, n_stackexchange, config="philosophy", level_prior=P(general_public=1.0),
           desc="philosophy.stackexchange: philosophy of mind / AI consciousness questions (filtered)"),
    Source("ml_qa_mjphayes", "mjphayes/machine_learning_questions", n_qa("question", "answer"),
           meta_rule=ACCEPT, level_prior=P(undergrad_non_ai=0.5, ai_student=0.5),
           desc="636 short ML concept Q&A"),
    Source("ai_qa_harikrishnan", "Harikrishnan46624/AI_QA_Data", n_qa("question", "answer"),
           meta_rule=ACCEPT, level_prior=P(undergrad_non_ai=0.5, ai_student=0.5),
           desc="1.1k AI/NLP/CV concept Q&A"),
    Source("ds_qa_soufyane", "soufyane/DATA_SCIENCE_QA", n_qa("Question", "Answer"),
           meta_rule=ACCEPT, level_prior=P(undergrad_non_ai=0.5, ai_student=0.5),
           desc="1k data-science interview Q&A"),
    Source("ml_qa_prsdm", "prsdm/Machine-Learning-QA-dataset", n_qa("Question", "Answer"),
           meta_rule=ACCEPT, level_prior=P(ai_student=1.0), desc="101 ML Q&A"),
    Source("so_python_dsml", "RazinAleks/SO-Python_QA-Data_Science_and_Machine_Learning_class", n_razin,
           level_prior=P(ai_student=0.6, researcher_non_ai=0.4),
           desc="StackOverflow Python Q&A tagged 'Data Science and ML' (the tag is noisy: e.g. MATLAB/Django "
                "questions) -> keyword + zero-shot decide"),
    Source("ml_qa_collection", "win-wang/Machine_Learning_QA_Collection", n_gemma_text,
           level_prior=P(ai_student=1.0), desc="12k StackOverflow-style ML Q&A (gemma chat format)"),
    Source("ai_interview_kareem", "K-areem/AI-Interview-Questions", n_inst_text,
           level_prior=P(ai_student=1.0), desc="AI/SE interview Q&A ([INST] format; mixed topics -> filtered)"),
    Source("ai_history_timeline", "chronicle-ai-history/ai-history-timeline", n_ai_history_timeline,
           meta_rule=ACCEPT, level_prior=P(general_public=0.6, high_school=0.4),
           desc="Curated AI history milestones -> templated history Q&A"),
    Source("ai_history_llm101_zh", "woohello/llm101-ai-history-sft", lambda r: n_instr_io(r, task="qa"), meta_rule=ACCEPT, lang="zh",
           level_prior=P(undergrad_non_ai=0.5, general_public=0.5),
           desc="Chinese LLM/AI-history SFT pairs (optional, lang=zh)"),
    Source("arxiv_ml_abstracts", "CShorten/ML-ArXiv-Papers", n_arxiv_abstract, meta_rule=ACCEPT,
           level_prior=P(ai_researcher=1.0), max_rows=40000,
           desc="ML arXiv titles+abstracts -> 'what is paper X about' research-level Q&A"),
    Source("arxiv_qa_taesiri", "taesiri/arxiv_qa", n_qa("question", "answer"), enabled=False,
           level_prior=P(ai_researcher=1.0), pool="huge", max_scan=200000,
           desc="(disabled) paper Q&A whose questions say 'this paper' without the paper -> needs rewriting"),
    Source("arxiv_dl_code", "AlgorithmicResearchGroup/ArXivDLInstruct", n_arxivdl, meta_rule=rule_arxivdl,
           domain="ai_code", pool="huge", max_scan=200000, max_rows=60000,
           columns=["prompt", "description", "function", "repo", "file"],
           level_prior=P(ai_student=0.5, ai_researcher=0.5),
           desc="Functions from deep-learning research repos with instructions (torch/tf/jax code accepted by meta)"),

    # ======================= B. Level-tagged explanations (huge, audience meta) =======================
    Source("cosmo_stanford", "HuggingFaceTB/cosmopedia", n_cosmopedia, config="stanford", pool="huge",
           level_prior=prior_cosmopedia, max_scan=400000, columns=["prompt", "text", "audience", "format", "seed_data"],
           desc="Textbook units from Stanford course titles, audience ∈ {young_children, high_school, college, researchers}"),
    Source("cosmo_khanacademy", "HuggingFaceTB/cosmopedia", n_cosmopedia, config="khanacademy", pool="huge",
           level_prior=lambda r: {"middle_school": 0.4, "high_school": 0.6},
           columns=["prompt", "text", "audience", "format", "seed_data"],
           desc="Khan-Academy-style units (computing / statistics courses)"),
    Source("cosmo_web_v2", "HuggingFaceTB/cosmopedia", n_cosmopedia, config="web_samples_v2", pool="huge",
           level_prior=prior_cosmopedia, max_scan=400000, columns=["prompt", "text", "audience", "format", "seed_data"],
           desc="Web-seeded explanations, audience ∈ {general, college_students}"),
    Source("cosmo_v2_smollm", "HuggingFaceTB/smollm-corpus", n_cosmopedia, config="cosmopedia-v2", pool="huge",
           level_prior=prior_cosmopedia, max_scan=600000, columns=["prompt", "text", "audience", "format", "seed_data"],
           desc="Cosmopedia v2: audience ∈ {children, middle_school_students, college_students, general, ...}"),
    Source("eli5", "sentence-transformers/eli5", n_qa("question", "answer"), config="pair", pool="huge",
           level_prior=P(general_public=0.7, middle_school=0.3),
           desc="Reddit ExplainLikeImFive: lay explanations (AI subset by filter)"),
    Source("webinstruct", "TIGER-Lab/WebInstructSub", n_webinstruct, pool="huge", max_scan=600000,
           meta_rule=lambda r: None, level_prior=prior_webinstruct,
           desc="Web Q&A (socratic=high-school homework, stackexchange, math.SE)"),

    # ======================= C. General instruction / chat pools (huge, filtered) =======================
    Source("openhermes", "teknium/OpenHermes-2.5", n_openhermes, pool="huge", meta_rule=rule_openhermes,
           desc="1M GPT-4 instructions; category meta drops role-play/jokes/etc."),
    Source("tulu3_mix", "allenai/tulu-3-sft-mixture", n_messages, pool="huge", meta_rule=rule_tulu,
           desc="Tulu-3 SFT mixture; `source` meta drops safety/jailbreak/multilingual subsets"),
    Source("smoltalk", "HuggingFaceTB/smoltalk", n_messages, config="all", pool="huge", meta_rule=rule_smoltalk,
           desc="SmolTalk; `source` meta drops rewrite/summarize/function-calling subsets"),
    Source("ultrachat", "HuggingFaceH4/ultrachat_200k", n_messages, split="train_sft", pool="huge",
           desc="UltraChat multi-turn dialogues about the world / concepts"),
    Source("wildchat", "allenai/WildChat-1M", n_wildchat, pool="huge", meta_rule=rule_wildchat, max_scan=600000,
           level_prior=P(general_public=1.0),
           columns=["conversation", "language", "toxic", "redacted", "model", "country"],
           desc="Real user ChatGPT logs: what normal citizens ask about AI (English, non-toxic only)"),
    Source("dolly", "databricks/databricks-dolly-15k", n_dolly,
           meta_rule=lambda r: "reject" if r.get("category") == "creative_writing" else None,
           level_prior=P(general_public=1.0), desc="Human-written instructions"),
    Source("nemotron_science", "nvidia/Llama-Nemotron-Post-Training-Dataset", n_nemotron, config="SFT",
           split="science", pool="huge", max_scan=400000,
           reasoning=True,
           desc="Science Q&A with R1 reasoning traces (<think>) -> reasoning data for AI questions"),
    Source("nemotron_chat", "nvidia/Llama-Nemotron-Post-Training-Dataset", n_nemotron, config="SFT",
           split="chat", pool="huge", desc="General chat with reasoning traces"),
    Source("nemotron_code", "nvidia/Llama-Nemotron-Post-Training-Dataset", n_nemotron, config="SFT",
           split="code", pool="huge", domain="ai_code", max_scan=104597,
           reasoning=True,
           desc="Code problems with reasoning traces; ML/Python subset by filter"),
    Source("openthoughts", "open-thoughts/OpenThoughts-114k", n_openthoughts, config="metadata", pool="huge",
           domain="ai_math",
           meta_rule=lambda r: "reject" if r.get("domain") in ("biology", "chemistry", "puzzle") else None,
           reasoning=True,
           desc="R1 reasoning traces (math/code/science); domain meta + filter"),

    # ======================= D. Python / ML code SFT =======================
    Source("glaive_code", "glaiveai/glaive-code-assistant-v3", n_qa("question", "answer", "code"), pool="huge",
           domain="ai_code", max_scan=500000, level_prior=P(ai_student=1.0),
           desc="Code assistant Q&A (lots of numpy/pandas/sklearn/torch questions)"),
    Source("magicoder_oss", "ise-uiuc/Magicoder-OSS-Instruct-75K", n_qa("problem", "solution", "code"),
           domain="ai_code", meta_rule=lambda r: None if r.get("lang") == "python" else "reject",
           level_prior=P(ai_student=1.0), desc="OSS-seeded coding problems (python only via `lang` meta)"),
    Source("magicoder_evol", "ise-uiuc/Magicoder-Evol-Instruct-110K", n_qa("instruction", "response", "code"),
           domain="ai_code", level_prior=P(ai_student=1.0), desc="Evol-instruct coding"),
    Source("codefeedback", "m-a-p/CodeFeedback-Filtered-Instruction", n_qa("query", "answer", "code"),
           domain="ai_code", meta_rule=lambda r: None if (r.get("lang") or "").lower() == "python" else "reject",
           level_prior=P(ai_student=1.0), desc="Filtered code instructions (python only via `lang` meta)"),
    Source("self_oss_python", "bigcode/self-oss-instruct-sc2-exec-filter-50k", n_qa("instruction", "response", "code"),
           domain="ai_code", level_prior=P(ai_student=1.0), desc="Execution-filtered Python instructions"),
    Source("python_tested_143k", "Vezora/Tested-143k-Python-Alpaca", n_instr_io, domain="ai_code",
           level_prior=P(ai_student=0.5, high_school=0.5), reasoning=True,
           desc="Python alpaca, tested"),
    Source("python_18k", "iamtarun/python_code_instructions_18k_alpaca", n_instr_io, domain="ai_code",
           level_prior=P(high_school=0.5, ai_student=0.5), desc="Python alpaca 18k"),

    # ======================= E. RL: verifiable code =======================
    Source("kodcode_v1", "KodCode/KodCode-V1", n_kodcode, bucket="ai_rl_code", domain="ai_code", pool="huge",
           columns=["subset", "question_id", "question", "solution", "test", "test_info", "gpt_difficulty",
                    "gpt_pass_percentage"],
           level_prior=P(ai_student=1.0), desc="Python problems + pytest unit tests (subset meta: Docs/Package = library usage)"),
    Source("kodcode_rl10k", "KodCode/KodCode-Light-RL-10K", n_kodcode, bucket="ai_rl_code", domain="ai_code",
           columns=["subset", "question_id", "question", "solution", "test", "test_info", "gpt_difficulty",
                    "gpt_pass_percentage"],
           level_prior=P(ai_student=1.0), desc="RL-curated KodCode subset"),

    # ======================= F. RL: verifiable math (AI-relevant) =======================
    Source("deepmath_103k", "zwhe99/DeepMath-103K", n_deepmath, bucket="ai_rl_math", domain="ai_math",
           meta_rule=rule_deepmath, level_prior=prior_math_difficulty,
           columns=["question", "final_answer", "difficulty", "topic", "r1_solution_1"],
           reasoning=True,
           desc="Hard math with final answers; `topic` meta selects lin-alg/prob/stats/calculus/optimisation"),
    Source("numina_cot", "AI-MO/NuminaMath-CoT", n_numina, bucket="ai_rl_math", domain="ai_math", pool="huge",
           level_prior=prior_numina, columns=["source", "problem", "solution"],
           reasoning=True,
           desc="860k math CoT; boxed answer as ground truth; prob/stat/calculus subset by filter"),
    Source("openr1_math", "open-r1/OpenR1-Math-220k", n_openr1, config="default", bucket="ai_rl_math",
           domain="ai_math", meta_rule=rule_openr1, level_prior=P(high_school=0.5, ai_student=0.5),
           columns=["problem", "solution", "answer", "problem_type", "question_type", "source"],
           reasoning=True,
           desc="Verified math; problem_type meta drops geometry/number theory/puzzles"),
    Source("deepscaler", "agentica-org/DeepScaleR-Preview-Dataset", n_deepscaler, bucket="ai_rl_math",
           domain="ai_math", level_prior=P(high_school=0.5, ai_student=0.5),
           reasoning=True,
           desc="40k competition math with answers (AI-relevant subset by filter)"),
    Source("nemotron_math", "nvidia/Llama-Nemotron-Post-Training-Dataset", n_nemotron, config="SFT",
           split="math", bucket="ai_rl_math", domain="ai_math", pool="huge", max_scan=276536,
           reasoning=True,
           desc="Math with R1 reasoning; boxed answers -> RL ground truth"),

    # ======================= G. Evaluation (never train) =======================
    Source("eval_mmlu", "cais/mmlu", n_mmlu, config="all", split="test", bucket="ai_eval",
           meta_rule=rule_mmlu, level_prior=prior_mmlu, desc="MMLU: machine_learning (+ filtered CS/stats subjects)"),
    Source("eval_mmlu_pro", "TIGER-Lab/MMLU-Pro", n_mmlu_pro, split="test", bucket="ai_eval",
           meta_rule=rule_mmlu_pro, level_prior=P(ai_student=1.0),
           desc="MMLU-Pro: computer science/math/engineering, AI items by filter"),
    Source("eval_ds1000", "xlangai/DS-1000", n_ds1000, split="test", bucket="ai_eval", domain="ai_code",
           meta_rule=ACCEPT, level_prior=P(ai_student=1.0),
           desc="DS-1000: numpy/pandas/sklearn/torch/tf problems with execution tests"),
]


# --------------------------------------------------------------------------- #
# v0.2 sources: teacher reasoning traces, verifiable RL data, plain-language text
# --------------------------------------------------------------------------- #
_LETTER = re.compile(r"^\s*(?:\\text\{)?\(?([A-J])\)?\}?\s*$")


def _letter(x) -> str | None:
    m = _LETTER.match(str(x or ""))
    return m.group(1) if m else None


_OPTION_LINE = re.compile(r"^\s*\(?([A-J])[:.)]\s+\S", re.M)


def mcq_letter_from_answer(question: str, answer: str) -> str | None:
    """Answer letter of a lettered multiple-choice question, read from the final \\boxed{} of the reference
    answer (used when the dataset ships no expected answer; DeepSeek-R1 answers, so not human-verified)."""
    opts = _OPTION_LINE.findall(question or "")
    if len(opts) < 3 or opts[:3] != ["A", "B", "C"]:
        return None
    letter = _letter(last_boxed(answer or "") or "")
    return letter if letter in opts else None


def n_nemotron_v1(r):
    """nvidia/Nemotron-Post-Training-Dataset-v1: messages + reasoning on/off + metadata.expected_answer."""
    msgs = as_list(r.get("messages"))
    d = chat_to_qa(msgs)
    if not d:
        return None
    md = r.get("metadata")
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except ValueError:
            md = {}
    exp = (md or {}).get("expected_answer")
    from_answer = None if exp else mcq_letter_from_answer(d["question"], d["answer"])
    d["ground_truth"] = _letter(exp) or from_answer or (str(exp) if exp else None)
    d["meta"] = {"category": r.get("category"), "generator": r.get("generator"),
                 "reasoning_mode": r.get("reasoning"), "mcq": bool(_letter(exp) or from_answer),
                 "gt_from": "expected_answer" if exp else ("model_answer" if from_answer else None)}
    return d


def n_openscience2(r):
    q, out = r.get("input") or "", r.get("output") or ""
    reasoning, ans = split_think(out)
    if not q or not ans:
        return None
    exp = r.get("expected_answer")
    return {"question": q, "answer": ans, "reasoning": reasoning, "task": "chat",
            "ground_truth": _letter(exp) or exp, "meta": {"mcq": bool(_letter(exp)), "generator": "see card"}}


def n_smoltalk2(r):
    d = chat_to_qa(as_list(r.get("messages")))
    if d:
        d["meta"] = {"orig_source": r.get("source")}
    return d


def n_megascience(r):
    q, a = (r.get("question") or "").strip(), (r.get("answer") or "").strip()
    if not q or not a:
        return None
    return {"question": q, "answer": a, "ground_truth": r.get("reference_answer"), "task": "qa",
            "meta": {"subject": r.get("subject"), "orig_source": r.get("source")}}


def n_webinstruct_verified(r):
    q, a = (r.get("question") or "").strip(), (r.get("answer") or "").strip()
    if not q or not a:
        return None
    return {"question": q, "answer": a, "ground_truth": a, "task": "qa",
            "meta": {"category": r.get("category"), "answer_type": r.get("answer_type"),
                     "difficulty": r.get("difficulty")}}


def n_synthetic1(r):
    """PrimeIntellect/SYNTHETIC-1 verifiable_code: stdin/stdout or function tests in verification_info."""
    q = r.get("prompt") or ""
    vi = r.get("verification_info")
    if not q or not vi:
        return None
    reasoning, ans = split_think(r.get("llm_response") or "")
    ans = ans or r.get("gold_standard_solution")
    tests = vi if isinstance(vi, str) else json.dumps(vi)
    return {"question": q.strip(), "answer": ans or None, "reasoning": reasoning, "tests": tests, "task": "rl_code",
            "meta": {"task_type": r.get("task_type"), "test_format": "synthetic1_verification_info",
                     "score": r.get("score"), "language": (r.get("metadata") or {}).get("language")
                     if isinstance(r.get("metadata"), dict) else None}}


def n_simplewiki(r):
    """Simple English Wikipedia article -> plain-language 'what is X' answer (lead paragraphs, <= ~180 words)."""
    title, text = (r.get("title") or "").strip(), (r.get("text") or "").strip()
    if not title or len(text) < 300:
        return None
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]
    out, words = [], 0
    for p in paras:
        if words + len(p.split()) > 180 and out:
            break
        out.append(p)
        words += len(p.split())
    if words < 40:
        return None
    return {"question": f"What is {title}? Please explain it in simple words.", "answer": "\n\n".join(out),
            "task": "explain", "meta": {"title": title, "url": r.get("url")}}


SOURCES += [
    Source("nemotron_v1_stem", "nvidia/Nemotron-Post-Training-Dataset-v1", n_nemotron_v1, split="stem",
           pool="huge", max_scan=300000, added_in="v0.2", license="CC BY 4.0",
           columns=["messages", "reasoning", "metadata", "category", "generator"],
           level_prior=P(ai_student=0.6, ai_researcher=0.4),
           reasoning=True,
           desc="STEM Q&A with DeepSeek-R1-0528 reasoning traces; expected_answer -> RL multiple choice"),
    Source("openscience_reasoning2", "nvidia/OpenScienceReasoning-2", n_openscience2, pool="huge", domain="reasoning",
           added_in="v0.2",
           license="CC BY 4.0", level_prior=P(ai_student=0.7, ai_researcher=0.3),
           desc="Science MCQ with <think> traces + expected answer letter (SFT reasoning + RL MCQ)"),
    Source("smoltalk2_openthoughts3_think", "HuggingFaceTB/smoltalk2", n_smoltalk2, config="SFT",
           split="OpenThoughts3_1.2M_think", pool="huge", max_scan=300000, domain="reasoning", added_in="v0.2",
           level_prior=P(ai_student=1.0), columns=["messages", "source"],
           desc="OpenThoughts3 reasoning traces (math/code/science); AI subset -> ai_sft, rest -> reasoning_sft"),
    Source("megascience", "MegaScience/MegaScience", n_megascience, pool="huge", added_in="v0.2",
           license="CC BY-NC-SA 4.0",
           meta_rule=lambda r: None if (r.get("subject") or "") in ("cs", "math", "") else "reject",
           level_prior=P(undergrad_non_ai=0.5, ai_student=0.5),
           desc="Textbook-derived science Q&A with reference answers; cs/math subset by filter"),
    Source("webinstruct_verified", "TIGER-Lab/WebInstruct-verified", n_webinstruct_verified, added_in="v0.2",
           license="Apache 2.0", domain="ai_math",
           meta_rule=lambda r: None if r.get("category") in ("Computer Science", "Mathematics", "Other STEM") else "reject",
           level_prior=P(undergrad_non_ai=0.5, ai_student=0.5),
           desc="Questions with verified short answers (RL); CS/math by filter"),
    # the coding-only split of SYNTHETIC-1: the full repo is ~98 % verifiable_math, so a 1-shard budget
    # there yielded 11 coding rows out of 18k
    Source("synthetic1_code", "PrimeIntellect/verifiable-coding-problems", n_synthetic1, bucket="ai_rl_code",
           domain="ai_code", pool="huge", added_in="v0.2", license="Apache 2.0",
           level_prior=P(ai_student=1.0),
           desc="Verifiable coding problems with test cases (+ R1 solutions) for RL code"),
    Source("simplewiki_ai", "wikimedia/wikipedia", n_simplewiki, config="20231101.simple", pool="huge",
           added_in="v0.2", license="CC BY-SA 4.0", columns=["title", "text", "url"],
           level_prior=P(primary_school=0.5, middle_school=0.5),
           desc="Simple English Wikipedia AI/computing articles -> plain-language explanations for young learners"),
]

SOURCES_BY_NAME = {s.name: s for s in SOURCES}
