#!/usr/bin/env python3
"""Keep only documents that explain machine learning technically, using concept-family keywords.

Why not ai_qa_data/keywords.py (kw_core)?  That scorer measures "is this ABOUT AI": field names ("artificial
intelligence"), companies (OpenAI), history, ethics and applications all carry weight 3, and it reads only the first
4,000 characters. A news article about an AI start-up scores as high as a derivation of backpropagation.

This filter asks a different question: does the document USE several distinct ML concepts?

  * ~40 concept FAMILIES (gradient descent, loss functions, regularisation, attention, kernels, Bayesian inference,
    RL value functions, ...). Synonyms are one family, so "SGD" + "stochastic gradient descent" count once.
    Field names, companies, products, history, ethics and applications are deliberately NOT keywords.
  * ANCHOR families are unambiguous ML (backprop, softmax, overfitting, hidden layers, ...). Ambiguous families
    (entropy, Markov chains, eigenvalues, linear regression, tokenization, "language model", fine-tuning, ...) count
    toward the family total but cannot qualify a document on their own -> physics / statistics / crypto / AI-news
    pages that merely share words are rejected.
  * DENSITY: ML hits per 1,000 words (each family capped at --hit-cap hits), so a 20k-token paper or book that
    mentions three concepts in passing does not pass.
  * NEWS penalty: business/news markers (CEO, funding, announced, customers, ...) must stay below
    --max-news-ratio x the ML hits.
  * DEPTH markers (LaTeX, math symbols, equations, derivation phrases, pseudo-code, code) are recorded and can be
    required with --min-depth; they are not needed to pass by default.
  The whole document is scanned (up to --max-chars).

Default rule:  families >= 3  AND  anchor families >= 1  AND  density >= 2.0 hits / 1k words
               AND  news hits <= 0.5 x ML hits  AND  words >= 100
A document with one or two keyword families is never kept (--min-families cannot be set below 2).

Inputs: folders (searched recursively at any depth, symlinks followed, hidden folders such as .cache skipped) or
files of .parquet, or JSON lines (.jsonl / .ndjson /
.json, optionally .gz or .zst / .zstd). A JSON input's output is parquet with columns text, id, token_count (when
present), raw (the original line, filter mode) and tech_*. A folder that holds any parquet file is treated as a
parquet dataset: its JSON files are ignored as metadata. If an input holds none of these, the run stops before
creating anything and lists the file types it did find. Any error in a worker stops the whole run with the file name
and exit code 1; finished files are kept, so rerunning the same command resumes.

Usage
  # first, check what the script sees: file types, columns, and the keep rate of the first 2,000 docs; writes nothing
  python3 -m cpt_trainer.filter_technical --check /data/fineweb-edu/data /data/nemotron-cc-v2/High-Quality
  # filter: mirror the input tree, keeping the passing rows (all original columns + tech_* columns)
  python3 -m cpt_trainer.filter_technical cpt_data/corpus_ai_fineweb_large cpt_data/tech/fineweb --workers 6
  # score only: id + tech_* columns for EVERY row (small; use it to pick thresholds, then filter by join)
  python3 -m cpt_trainer.filter_technical cpt_data/corpus_ai_nemotron_large cpt_data/tech/nemotron_scores --mode score
  # explain one document (stdin or a file) / run the built-in checks
  python3 -m cpt_trainer.filter_technical --explain some_doc.txt
  python3 -m cpt_trainer.filter_technical --selftest

Logs: timestamped lines on stdout and in <out>/filter.log: file listing and total row count, a heartbeat every
--log-every seconds (docs done, kept, docs/s, ETA), per-file start / progress (--file-log-every) / done lines.
A file's output parquet appears only when that whole file is finished; progress before that is in the log.
Output: <out>/<input folder name>/<relative path>.parquet per input file, <out>/_stats/*.json (resume: finished files are
skipped), <out>/summary.json and <out>/summary.md (docs and tokens in/kept per top-level subset, reject reasons,
family-count histogram), and with --examples N a review file <out>/examples.jsonl of kept and near-miss documents.
Tokens use the `token_count` column when present (FineWeb-Edu, GPT-2 count ~3 % above Qwen3), else chars / 4.6.
Speed (~4x the plain-regex version, identical results): every regex alternative gets a "gate" of literal phrases
that any match must contain ("loss function", "q value", "validation set"); one Aho-Corasick pass (pyahocorasick)
over the lower-cased text with whitespace and hyphen runs collapsed finds the phrases present, and each family regex is built from
its present alternatives only (an absent one can never match). In filter mode a document is dropped as soon as it can
no longer reach the rule (too few candidate families, no anchor), so most documents never run a regex.
Install: pip install pyarrow pyahocorasick zstandard   (pyahocorasick is optional but gives most of the speed;
zstandard only for .zst inputs). Self-contained, no project imports, so the file can be copied elsewhere.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool

# --------------------------------------------------------------------------------------------------------------- #
# Concept families: name -> (anchor, lower-case patterns, case-sensitive patterns)
# Lower-case patterns are matched on text.lower() with word-ish boundaries; case-sensitive ones on the original text.
# --------------------------------------------------------------------------------------------------------------- #
S = r"[\s-]+"      # separator: space(s) or hyphen
O = r"[\s-]?"      # optional separator
FAMILIES: dict[str, tuple[bool, list[str], list[str]]] = {
    # ---------------- anchors: unambiguous ML ----------------
    "backprop": (True, [rf"back{O}propagat\w*", r"backprop"], []),
    "gradient_descent": (True, [rf"gradient{S}(?:descent|ascent)", rf"steepest{S}descent"], [r"SGD"]),
    "loss_function": (True, [rf"loss{S}functions?", rf"cross{O}entropy", rf"negative{S}log{O}likelihood", rf"hinge{S}loss",
                             rf"log{S}loss", rf"mean{S}squared{S}errors?", rf"(?:training|validation|test|reconstruction){S}loss(?:es)?",
                             rf"loss{S}(?:curves?|landscapes?|surfaces?)"], [r"MSE", r"NLL"]),
    "activation": (True, [rf"activation{S}functions?", rf"soft{O}max", r"relu", rf"leaky{S}relu", r"gelu", r"swish",
                          rf"sigmoid{S}(?:activations?|functions?|units?|neurons?|layers?)", rf"tanh{S}(?:activations?|functions?|units?|layers?)"], []),
    "network_structure": (True, [rf"hidden{S}(?:layers?|units?|neurons?|states?)", rf"(?:fully{S}connected|dense|input|output|linear){S}layers?",
                                 rf"multi{O}layer{S}perceptrons?", r"perceptrons?", rf"feed{O}forward{S}(?:neural{S})?networks?",
                                 rf"neural{S}network{S}(?:weights|architectures?|layers?)", rf"weights{S}and{S}bias(?:es)?"], [r"MLPs?"]),
    "overfitting": (True, [rf"over{O}fit\w*", rf"under{O}fit\w*", rf"generali[sz]ation{S}(?:error|gap|bounds?)", rf"bias{O}[\s-]{O}variance"], []),
    "regularization": (True, [rf"regulari[sz]ation{S}(?:terms?|parameters?|strength|penalt(?:y|ies)|techniques?|methods?)",
                              rf"l[12]{S}(?:regulari[sz]ation|penalty|norm)", rf"weight{S}decay", rf"drop{O}out{S}(?:layers?|rates?|probability|regulari[sz]ation)",
                              rf"early{S}stopping", rf"(?:ridge|lasso){S}regression", rf"elastic{S}net"], []),
    "convolutional": (True, [rf"convolutional{S}(?:neural{S}networks?|layers?|filters?|kernels?|networks?|nets?)", rf"(?:max|average|global){S}pooling",
                             rf"pooling{S}layers?", rf"feature{S}maps?", rf"receptive{S}fields?"], [r"CNNs?", r"ConvNets?", r"ResNets?(?:-\d+)?"]),
    "recurrent": (True, [rf"recurrent{S}neural{S}networks?", rf"long{S}short{O}term{S}memory", rf"seq(?:uence)?{O}(?:2|to){O}seq(?:uence)?",
                         rf"gated{S}recurrent", rf"backpropagation{S}through{S}time"], [r"RNNs?", r"LSTMs?", r"GRUs?"]),
    "attention": (True, [rf"self{O}attention", rf"(?:multi{O}head|cross|scaled{S}dot{O}product|causal|masked){S}attention",
                         rf"attention{S}(?:mechanisms?|heads?|weights|scores|layers?|matri(?:x|ces)|maps?)",
                         rf"transformer{S}(?:architectures?|models?|blocks?|layers?|encoders?|decoders?)", rf"positional{S}(?:encodings?|embeddings?)",
                         rf"(?:query|queries),?{S}keys?,?{S}(?:and{S})?values?", rf"encoder{O}decoder"], []),
    "embeddings": (True, [rf"word{O}(?:2vec|embeddings?)", rf"(?:token|sentence|text|vector){S}embeddings?",
                          rf"embedding{S}(?:vectors?|layers?|spaces?|dimensions?|matri(?:x|ces))", rf"glove{S}(?:vectors|embeddings)",
                          rf"latent{S}(?:spaces?|representations?|vectors?)"], []),
    "optimizer": (True, [rf"learning{S}rates?", rf"adam(?:w)?{S}optimi[sz]ers?", r"rms{O}prop", r"adagrad", r"adadelta", r"adamw",
                         rf"momentum{S}(?:terms?|coefficients?|sgd|optimi[sz]ers?)", rf"cosine{S}annealing"], []),
    "hyperparameters": (True, [rf"hyper{O}parameters?", rf"batch{S}sizes?", rf"mini{O}batch(?:es)?", rf"(?:training|number{S}of){S}epochs",
                               rf"grid{S}search", rf"bayesian{S}optimi[sz]ation"], []),
    "eval_metrics": (True, [rf"cross{O}validation", rf"k{O}fold", rf"train(?:ing)?{O}[/\s-]{O}(?:test|validation){S}splits?",
                            rf"(?:validation|held{O}out|dev(?:elopment)?){S}(?:sets?|data(?:sets?)?)", rf"confusion{S}matri(?:x|ces)",
                            rf"precision{S}and{S}recall", rf"f1{O}scores?", rf"roc{S}curves?", rf"area{S}under{S}the{S}(?:roc{S})?curve",
                            rf"mean{S}average{S}precision"], [r"AUC", r"ROC"]),
    "classifiers": (True, [rf"logistic{S}regression", rf"(?:softmax|multinomial){S}regression", rf"linear{S}classifiers?", rf"naive{S}bayes",
                           rf"k{O}nearest{S}neighbou?rs?", rf"decision{S}boundar(?:y|ies)", rf"(?:binary|multi{O}class|multi{O}label){S}classifi(?:er|ers|cation)",
                           r"classifiers?"], [r"k-?NN"]),
    "svm_kernels": (True, [rf"support{S}vector{S}(?:machines?|classifiers?|regression)", rf"support{S}vectors", rf"kernel{S}(?:trick|methods?|functions?|matri(?:x|ces)|ridge|pca|svm)",
                           rf"(?:rbf|radial{S}basis|polynomial|gaussian){S}kernels?", rf"max(?:imum)?{O}margin", rf"soft{S}margin"], [r"SVMs?"]),
    "trees_ensembles": (True, [rf"random{S}forests?", rf"gradient{S}boost(?:ing|ed)", r"xgboost", r"lightgbm", r"catboost", r"adaboost", r"bagging",
                               rf"decision{S}trees?", rf"gini{S}impurity", rf"information{S}gain", rf"weak{S}learners?", rf"ensemble{S}(?:methods?|learning|models?)"], []),
    "clustering": (True, [rf"k{O}means", rf"(?:hierarchical|agglomerative|spectral|density{O}based){S}clustering", rf"clustering{S}(?:algorithms?|methods?)",
                          r"dbscan", rf"gaussian{S}mixture{S}models?", rf"cluster{S}(?:centroids?|centers?|assignments?)"], [r"GMMs?"]),
    "latent_variational": (True, [rf"expectation{O}maximi[sz]ation", rf"variational{S}(?:inference|bayes|lower{S}bound|distributions?)",
                                  rf"evidence{S}lower{S}bound", rf"latent{S}dirichlet{S}allocation", rf"reparameteri[sz]ation{S}trick"], [r"ELBO"]),
    "reinforcement": (True, [rf"q{O}learning", rf"q{O}values?", rf"q{O}functions?", rf"policy{S}(?:gradients?|networks?|iteration|optimi[sz]ation|evaluation)",
                             rf"value{S}(?:functions?|iteration)", rf"bellman{S}(?:equations?|operators?|backups?)", rf"markov{S}decision{S}process(?:es)?",
                             rf"temporal{O}difference", rf"actor{O}critic", rf"epsilon{O}greedy", rf"exploration{S}(?:vs\.?|versus|and){S}exploitation",
                             rf"reward{S}(?:functions?|signals?|shaping|models?)", rf"multi{O}armed{S}bandits?", rf"discount{S}factors?"], [r"MDPs?", r"PPO", r"DQNs?"]),
    "generative_models": (True, [rf"generative{S}adversarial{S}networks?", rf"(?:generator|discriminator){S}networks?", rf"(?:variational{S})?auto{O}encoders?",
                                 rf"denoising{S}diffusion", rf"diffusion{S}models?", rf"normali[sz]ing{S}flows?", rf"score{S}matching", rf"mode{S}collapse"],
                          [r"GANs?", r"VAEs?"]),
    "feature_engineering": (True, [rf"feature{S}(?:engineering|extraction|selection|vectors?|scaling|importances?|spaces?)", rf"one{O}hot{S}encod\w+",
                                   rf"(?:min{O}max|z{O}score){S}(?:normali[sz]ation|scaling)", rf"(?:input|categorical|numerical){S}features"], []),
    "nlp_methods": (True, [rf"tf{O}idf", rf"bag{S}of{S}words", rf"n{O}grams?", rf"part{O}of{O}speech{S}tagg\w+", rf"named{S}entity{S}recognition",
                           rf"word{S}sense{S}disambiguation", rf"dependency{S}pars\w+", r"lemmati[sz]\w+", rf"bleu{S}scores?", rf"sequence{S}labell?ing"], []),
    "vision_methods": (True, [rf"object{S}detection", rf"bounding{S}box(?:es)?", rf"(?:semantic|instance|panoptic|image){S}segmentation",
                              rf"image{S}classification", rf"non{O}max(?:imum)?{S}suppression", rf"intersection{S}over{S}union", rf"(?:data|image){S}augmentation",
                              rf"anchor{S}box(?:es)?"], [r"IoU", r"YOLO\w*", r"R-CNN"]),
    "decoding": (True, [rf"beam{S}search", rf"greedy{S}decoding", rf"top{O}[kp]{S}sampling", rf"nucleus{S}sampling",
                        rf"(?:sampling{S})?temperature{S}(?:parameter|sampling|scaling)", rf"kv{S}cache", r"logits?"], []),
    "search_planning": (True, [rf"a\*{S}(?:search|algorithm)", rf"best{O}first{S}search", rf"(?:breadth|depth){O}first{S}search",
                               rf"(?:admissible{S})?heuristic{S}(?:functions?|search)", r"minimax", rf"alpha{O}beta{S}pruning", rf"constraint{S}satisfaction",
                               rf"state{O}space{S}search", rf"search{S}trees?"], []),
    "graph_nn": (True, [rf"graph{S}neural{S}networks?", rf"message{S}passing", rf"graph{S}(?:convolution\w*|attention{S}networks?)", rf"node{S}embeddings?"],
                 [r"GNNs?", r"GCNs?"]),
    "training_tricks": (True, [rf"batch{S}normali[sz]ation", rf"layer{S}normali[sz]ation", rf"(?:vanishing|exploding){S}gradients?", rf"gradient{S}clipping",
                               rf"weight{S}initiali[sz]ation", rf"(?:xavier|kaiming|glorot|he){S}initiali[sz]ation", rf"residual{S}(?:connections?|blocks?)",
                               rf"skip{S}connections?"], [r"BatchNorm", r"LayerNorm"]),
    # ---------------- ambiguous: count toward the total, cannot qualify a document alone ----------------
    "tokenization": (False, [r"tokeni[sz](?:er|ers|ation|ed|ing)", rf"byte{S}pair{S}encoding", rf"sub{O}word{S}(?:units?|tokens?)", rf"word{O}piece",
                             rf"sentence{O}piece", rf"vocabulary{S}sizes?"], [r"BPE"]),
    "language_modeling": (False, [rf"language{S}models?", rf"next{S}(?:token|word){S}prediction", rf"autoregressive{S}(?:models?|generation|decoding)",
                                  rf"masked{S}language{S}model\w*", rf"context{S}(?:windows?|lengths?)", r"perplexity"], [r"LLMs?", r"GPT-?\d\w*", r"BERT"]),
    "training_procedure": (False, [rf"fine{O}tun(?:e|ed|es|ing)", rf"pre{O}train(?:ed|ing)", rf"transfer{S}learning", rf"knowledge{S}distillation",
                                   rf"low{O}rank{S}adaptation", rf"instruction{S}tuning", rf"training{S}(?:loops?|runs?|steps?|objectives?|procedures?)"],
                           [r"LoRA", r"RLHF"]),
    "learning_paradigms": (False, [rf"(?:un|semi{O}|self{O})?supervised{S}learning", rf"reinforcement{S}learning", rf"(?:few|zero|one){O}shot{S}learning",
                                   rf"(?:contrastive|representation|meta|active|online|federated|multi{O}task){S}learning"], []),
    "data_splits": (False, [rf"(?:training|test|testing){S}(?:data(?:sets?)?|sets?|examples|samples|instances)", rf"labell?ed{S}(?:data|examples|datasets?)",
                            rf"unlabell?ed{S}data", rf"ground{O}truth", rf"class{S}imbalance"], []),
    "dim_reduction": (False, [rf"principal{S}components?", rf"dimensionality{S}reduction", rf"singular{S}value{S}decomposition", r"t-?sne",
                              rf"manifold{S}learning"], [r"PCA", r"SVD", r"UMAP"]),
    "bayesian": (False, [rf"bayes(?:'|’)?s?{S}(?:theorem|rule|formula)", rf"bayesian{S}(?:inference|networks?|models?|methods?|statistics|approach|updating)",
                         rf"(?:posterior|prior){S}(?:distributions?|probabilit(?:y|ies)|beliefs?)", rf"likelihood{S}functions?",
                         rf"maximum{S}(?:likelihood|a{S}posteriori)", rf"conjugate{S}priors?", rf"marginal{S}likelihood", rf"latent{S}variables?"],
                 [r"MLE", r"MAP"]),
    "info_theory": (False, [r"(?<!cross-)(?<!cross )(?<!cross)entropy", rf"mutual{S}information", rf"kl{S}divergence", rf"kullback{O}leibler",
                            rf"information{S}theory"], []),
    "markov_sampling": (False, [rf"markov{S}(?:chains?|process(?:es)?|property|models?|random{S}fields?)", rf"hidden{S}markov{S}models?", rf"monte{S}carlo",
                                rf"gibbs{S}sampl\w+", rf"metropolis{O}hastings", rf"importance{S}sampling"], [r"MCMC", r"HMMs?"]),
    "linear_algebra": (False, [r"eigen(?:values?|vectors?|decompositions?)", rf"matrix{S}(?:multiplications?|factori[sz]ations?|inversion|decompositions?)",
                               rf"(?:dot|inner){S}products?", r"jacobians?", r"hessians?", rf"covariance{S}matri(?:x|ces)", rf"(?:weight|parameter){S}matri(?:x|ces)",
                               rf"vector{S}spaces?", rf"orthogonal{S}(?:matri(?:x|ces)|projections?|vectors?)", r"tensors?"], []),
    "calculus_optimization": (False, [rf"partial{S}derivatives?", rf"chain{S}rule", rf"gradients?{S}(?:of|with{S}respect{S}to)", rf"convex(?:ity)?{S}(?:functions?|optimi[sz]ation|problems?|sets?)",
                                      rf"(?:local|global){S}(?:minim(?:um|a)|optim(?:um|a))", rf"saddle{S}points?", rf"objective{S}functions?",
                                      rf"lagrange{S}multipliers?", rf"convergence{S}(?:rates?|guarantees?)"], []),
    "probability_stats": (False, [rf"random{S}variables?", rf"probability{S}(?:distributions?|density|mass)",
                                  rf"(?:gaussian|normal|bernoulli|binomial|poisson|multinomial|categorical|uniform){S}distributions?", rf"expected{S}values?",
                                  rf"standard{S}deviations?", rf"conditional{S}probabilit(?:y|ies)", r"i\.i\.d\.?", rf"identically{S}distributed"], []),
    "regression_stats": (False, [rf"linear{S}regression", rf"least{S}squares", rf"regression{S}(?:models?|coefficients?|analysis|lines?)",
                                 rf"r{O}squared", rf"coefficient{S}of{S}determination"], []),
    "frameworks": (False, [r"pytorch", r"tensorflow", r"keras", rf"scikit{O}learn", r"sklearn", rf"hugging{O}face{S}transformers"], [r"JAX"]),
}
# ML code: matched without word boundaries on lower-cased text; anchor.
CODE_PATTERNS = [r"import\s+(?:torch|tensorflow|keras|sklearn|jax)\b", r"from\s+(?:torch|tensorflow|keras|sklearn|transformers)[\s.]",
                 r"\bnn\.(?:module|linear|conv[12]d|lstm|gru|sequential|relu|crossentropyloss|mseloss)", r"\bmodel\.(?:fit|predict|train|eval|compile)\(",
                 r"\boptimizer\.(?:step|zero_grad)\(", r"\bloss\.backward\(", r"\btorch\.(?:tensor|randn|zeros|nn|optim|no_grad)", r"\btf\.(?:keras|data|nn)\.",
                 r"\.fit\(\s*x", r"train_test_split\("]

# Depth markers (original text): each TYPE counts once.
DEPTH_PATTERNS = {
    "latex": r"\$[^$\n]{1,200}\$|\\(?:frac|sum|prod|nabla|partial|mathbf|mathcal|theta|alpha|lambda|sigma|log|exp)\b|\\begin\{(?:equation|align)",
    "math_symbols": r"[θαβγδλσμηεπΣ∑∏∂∇∈∉≤≥≈∝→⊂∀∃‖]",
    "equation": r"(?<![\w])(?:[a-zA-Z](?:_\w+)?|[a-zA-Z]\([a-z, ]+\))\s*=\s*[-\w(.]+\s*[-+*/^·]\s*[\w(]",
    "derivation": r"(?i)\b(?:we\s+(?:derive|obtain|define|minimi[sz]e|maximi[sz]e|compute)|with\s+respect\s+to|let\s+[a-z]\s+(?:be|denote)|"
                  r"where\s+[a-z](?:_\w+)?\s+(?:is|denotes)|it\s+follows\s+that|subject\s+to|taking\s+the\s+derivative)\b",
    "pseudo_code": r"(?i)\b(?:algorithm\s+\d|pseudo-?code|for\s+each\s+\w+\s+(?:in|do)\b|repeat\s+until|end\s+for\b)",
    "code": r"```|\bdef\s+\w+\(|^\s*import\s+\w+|\bfor\s+\w+\s+in\s+range\(|\bprint\(|\breturn\s+\w",
}
NEWS_RE = re.compile(r"(?<![\w-])(?:ceo|cto|founders?|co-?founders?|start-?ups?|funding|raised|investors?|venture\s+capital|valuation|press\s+release|"
                     r"announc(?:e|ed|es|ement|ements)|launch(?:ed|es)?|customers?|revenue|partnerships?|stock|shares|market\s+(?:cap|share)|"
                     r"billion|subscribers?|pricing|enterprises?|vendors?|webinars?|sign\s+up|newsletter|sponsored)(?![\w-])")


def _split_alts(p: str) -> list[str]:
    """Split a pattern at its top-level '|' (outside groups and [...] classes)."""
    out, depth, cls, start, i = [], 0, False, 0, 0
    while i < len(p):
        c = p[i]
        if c == "\\":
            i += 2
            continue
        if cls:
            cls = c != "]"
        elif c == "[":
            cls = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "|" and depth == 0:
            out.append(p[start:i]); start = i + 1
        i += 1
    return out + [p[start:]]


_MAX_VARIANTS = 64


def _quant(alt: str, j: int) -> tuple[str, int]:
    """Quantifier at alt[j]: ('opt' | 'rep' | '', index after it)."""
    if j < len(alt) and alt[j] in "?*":
        return "opt", j + 1 + (alt[j + 1:j + 2] == "?")
    if j < len(alt) and alt[j] == "+":
        return "rep", j + 1
    if j < len(alt) and alt[j] == "{":
        k = alt.index("}", j)
        return ("opt" if alt[j + 1:k].split(",")[0] in ("", "0") else "rep"), k + 1
    return "", j


def _expand(alt: str):
    """All literal skeletons of a regex alternative, as a list of variants; each variant is a list of text chunks
    that every match of that variant contains in order (chunks are separated where the regex has a variable part).
    S ([\\s-]+) becomes one space, O ([\\s-]?) forks into "" and " ", a required group forks into its branches, an
    optional atom breaks the chunk. Returns None if there are more than _MAX_VARIANTS variants."""
    var = [[""]]

    def add(s):
        for v in var:
            v[-1] += s

    def brk():
        for v in var:
            if v[-1]:
                v.append("")

    def fork(options):
        nonlocal var
        new = [v[:-1] + [v[-1] + o[0]] + o[1:] for v in var for o in options]
        if len(new) > _MAX_VARIANTS:
            raise OverflowError
        var = new
    i = 0
    while i < len(alt):
        if alt.startswith(S, i):
            add(" "); i += len(S); continue
        if alt.startswith(O, i):
            fork([[""], [" "]]); i += len(O); continue
        c = alt[i]
        if c == "\\":
            atom_lit, j = (alt[i + 1] if not alt[i + 1].isalnum() else None), i + 2
        elif c == "[":
            j = i + 1
            while alt[j] != "]":
                j += 2 if alt[j] == "\\" else 1
            atom_lit, j = None, j + 1
        elif c == "(":
            depth, j = 0, i
            while True:
                if alt[j] == "\\":
                    j += 2; continue
                if alt[j] == "(":
                    depth += 1
                elif alt[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            inner, j = alt[i + 1:j], j + 1
            q, j = _quant(alt, j)
            if inner.startswith(("?<", "?=", "?!")):        # look-around: consumes nothing
                i = j; continue
            if q == "opt":
                brk(); i = j; continue
            inner = inner[2:] if inner.startswith("?:") else inner
            opts = []
            for b in _split_alts(inner):
                sub = _expand(b)
                if sub is None:
                    raise OverflowError
                opts += sub
            fork(opts)
            if q == "rep":
                brk()
            i = j; continue
        elif c in ".^$":
            atom_lit, j = (None if c == "." else ""), i + 1
        else:
            atom_lit, j = c, i + 1
        q, j = _quant(alt, j)
        if atom_lit is None or q == "opt":
            brk()
        else:
            add(atom_lit)
            if q == "rep":
                brk()
        i = j
    return var


def _required(alt: str, min_len: int = 3, join_sep: bool = True) -> list[tuple[str, ...]] | None:
    """Gate of one alternative: any-of AND-sets of literal chunks (one set per skeleton variant). None = cannot be
    gated (a variant has no chunk of >= min_len characters, or too many variants): the alternative always runs."""
    try:
        var = _expand(alt)
    except (OverflowError, IndexError, ValueError):
        return None
    out = []
    for v in var:
        chunks = tuple(sorted({c.strip() for c in v if len(c.strip()) >= min_len}))
        if not chunks:
            return None
        out.append(chunks)
    return sorted(set(out))


# literal triggers of the code family (its patterns are matched without word boundaries)
CODE_GATE = ("torch", "tensorflow", "keras", "sklearn", "jax", "transformers", "nn.", "model.fit", "model.predict",
             "model.train", "model.eval", "model.compile", "optimizer.", "loss.backward", "tf.", ".fit(", "train_test_split")
_NORM = re.compile(r"[\s-]+")      # lower-case gate phrases are searched in text with every [\s-]+ run -> one space

# Aho-Corasick (pyahocorasick) finds every gate literal in one pass; without it, `literal in text` per literal.
try:
    import ahocorasick as _ahoc
except ImportError:
    _ahoc = None
ENGINE = "aho-corasick (pyahocorasick)" if _ahoc is not None else "substring scan (pip install pyahocorasick for a faster run)"


class _Lits:
    """Literal table + automaton for one text view (normalised lower-case, or original case)."""

    def __init__(self):
        self.ids, self.lits, self.auto = {}, [], None

    def sets(self, req):
        """AND-sets of literals -> tuple of frozensets of ids (None stays None = always open)."""
        if req is None:
            return None
        out = []
        for st in req:
            for x in st:
                if x not in self.ids:
                    self.ids[x] = len(self.lits); self.lits.append(x)
            out.append(frozenset(self.ids[x] for x in st))
        return tuple(out)

    def build(self):
        if _ahoc is not None and self.lits:
            self.auto = _ahoc.Automaton()
            for x, i in self.ids.items():
                self.auto.add_word(x, i)
            self.auto.make_automaton()

    def present(self, text) -> set:
        if self.auto is not None:
            return {i for _, i in self.auto.iter(text)}
        return {i for i, x in enumerate(self.lits) if x in text}


def _compile():
    """Per family: its alternatives (lower-case and case-sensitive) with their literal gates. At run time only the
    alternatives whose gate is present are joined into the family regex: an absent alternative can never match, so
    dropping it leaves every match (and every hit count) exactly as with the full regex."""
    lo_l, up_l = _Lits(), _Lits()
    fams = []
    for name, (anchor, low, cs) in FAMILIES.items():
        la = [x for p in low for x in _split_alts(p)]
        ua = [x for p in cs for x in _split_alts(p)]
        fams.append({"name": name, "anchor": anchor, "code": False,
                     "lo": la, "lo_g": [lo_l.sets(_required(x, 3)) for x in la],
                     "up": ua, "up_g": [up_l.sets(_required(x, 2)) for x in ua]})
    fams.append({"name": "ml_code", "anchor": True, "code": True, "lo": ["|".join(CODE_PATTERNS)],
                 "lo_g": [lo_l.sets([(x,) for x in CODE_GATE])], "up": [], "up_g": []})
    lo_l.build(); up_l.build()
    for side, tbl in (("lo", lo_l), ("up", up_l)):     # literal id -> families it can open
        tbl.fam_of = {}
        for k, f in enumerate(fams):
            for g in f[side + "_g"]:
                for st in (g or ()):
                    for i in st:
                        tbl.fam_of.setdefault(i, set()).add(k)
    always = {k for k, f in enumerate(fams) if any(g is None for g in f["lo_g"] + f["up_g"])}
    depth = {k: re.compile(v, re.MULTILINE) for k, v in DEPTH_PATTERNS.items()}
    return fams, lo_l, up_l, always, depth


_FAMS, _LO, _UP, _ALWAYS, _DEPTH = _compile()
ANCHORS = sorted(f["name"] for f in _FAMS if f["anchor"])
AMBIGUOUS = sorted(f["name"] for f in _FAMS if not f["anchor"])
_ANCHOR_SET = frozenset(ANCHORS)
_RX: dict = {}      # (family, side, open alternatives) -> compiled regex


def _rx(k: int, side: str, idx: tuple):
    key = (k, side, idx)
    r = _RX.get(key)
    if r is None:
        f = _FAMS[k]
        body = "|".join(f[side][i] for i in idx)
        if f["code"]:
            r = re.compile(body)
        elif side == "lo":
            r = re.compile(r"(?<![\w-])(?:" + body + r")(?!\w)")
        else:
            r = re.compile(r"(?<![\w-])(?:" + body + r")(?![\w-])")
        _RX[key] = r
    return r


def _open_alts(gates, present) -> tuple:
    return tuple(i for i, g in enumerate(gates) if g is None or any(st <= present for st in g))


def _count(rx, text, cap):
    n = 0
    for _ in rx.finditer(text):
        n += 1
        if n >= cap:
            break
    return n


def analyze(text: str, hit_cap: int = 10, max_chars: int = 300_000, need: tuple[int, int] | None = None) -> dict:
    """Score one document. Returns families found, hit counts, density, depth and news markers.

    1. One Aho-Corasick pass finds the gate phrases present (lower-case text with [\\s-]+ collapsed to one space;
       acronyms on the original text).
    2. A family is a candidate if one of its alternatives has all its phrases present. With
       need=(min_families, min_anchors), a document whose candidates cannot reach the rule is returned at once with
       prefilter=True: it could never pass, so filtering results are exact while its regexes are skipped.
    3. Each candidate family's regex is built from its present alternatives only (cached) and counts the hits."""
    text = text[:max_chars] if max_chars else text
    low = text.lower()
    words = max(1, len(text.split()))
    p_lo = _LO.present(" ".join(low.replace("-", " ").split()))     # == _NORM.sub(" ", low), ~3x faster
    p_up = _UP.present(text) if _UP.lits else set()
    touched = set(_ALWAYS)
    for i in p_lo:
        touched |= _LO.fam_of.get(i, set())
    for i in p_up:
        touched |= _UP.fam_of.get(i, set())
    cand = []
    for k in sorted(touched):
        f = _FAMS[k]
        olo = _open_alts(f["lo_g"], p_lo) if f["lo"] else ()
        oup = _open_alts(f["up_g"], p_up) if f["up"] else ()
        if olo or oup:
            cand.append((k, olo, oup))
    if need is not None:
        nf, na = len(cand), sum(1 for c in cand if _FAMS[c[0]]["anchor"])
        if nf < need[0] or na < need[1]:
            return {"families": nf, "anchors": na, "hits": 0, "words": words, "density": 0.0, "news": 0, "depth": 0,
                    "family_list": [], "depth_list": [], "fam_hits": {}, "prefilter": True}
    fam_hits = {}
    cand.sort(key=lambda c: not _FAMS[c[0]]["anchor"])          # anchors first: "no anchor" is decided early
    n_anchor_cand = sum(1 for c in cand if _FAMS[c[0]]["anchor"])
    found_anchor = 0
    for pos, (k, olo, oup) in enumerate(cand):
        n = _count(_rx(k, "lo", olo), low, hit_cap) if olo else 0
        if oup and n < hit_cap:
            n += _count(_rx(k, "up", oup), text, hit_cap - n)
        if n:
            fam_hits[_FAMS[k]["name"]] = n
            found_anchor += _FAMS[k]["anchor"]
        if need is not None:                                    # exact early exit: the rule can no longer be met
            left = len(cand) - pos - 1
            if len(fam_hits) + left < need[0] or (pos + 1 >= n_anchor_cand and found_anchor < need[1]):
                return {"families": len(fam_hits) + left, "anchors": found_anchor, "hits": 0, "words": words, "density": 0.0,
                        "news": 0, "depth": 0, "family_list": [], "depth_list": [], "fam_hits": {}, "prefilter": True}
    hits = sum(fam_hits.values())
    anchors = [f for f in fam_hits if f in _ANCHOR_SET]
    # depth and news markers only matter for documents that can still pass; 0 otherwise
    may_pass = len(fam_hits) >= (need[0] if need else 2) and (need is None or len(anchors) >= need[1])
    depth = [k for k, rx in _DEPTH.items() if rx.search(text)] if may_pass else []
    news = _count(NEWS_RE, low, 1000) if may_pass else 0
    return {"families": len(fam_hits), "anchors": len(anchors), "hits": hits, "words": words,
            "density": round(1000.0 * hits / words, 3), "news": news, "depth": len(depth),
            "family_list": sorted(fam_hits), "depth_list": depth, "fam_hits": fam_hits}



def decide(r: dict, a) -> tuple[bool, str]:
    """Apply the rule; returns (keep, first failing reason or 'kept')."""
    if r["words"] < a.min_words:
        return False, "too_short"
    if r["families"] < a.min_families:
        return False, f"families<{a.min_families}"
    if r["anchors"] < a.min_anchors:
        return False, "no_anchor"
    if r["density"] < a.min_density:
        return False, "low_density"
    if r["news"] > a.max_news_ratio * r["hits"]:
        return False, "news"
    if r["depth"] < a.min_depth:
        return False, "no_depth"
    return True, "kept"


# --------------------------------------------------------------------------------------------------------------- #
# Corpus processing
# --------------------------------------------------------------------------------------------------------------- #
_A = None
_CNT = None          # shared counters: docs, kept, files_started, files_done
_LOGF = None

PARQUET_EXT = (".parquet",)
JSON_EXT = (".jsonl", ".jsonl.gz", ".jsonl.zst", ".jsonl.zstd", ".ndjson", ".ndjson.gz", ".ndjson.zst",
            ".json", ".json.gz", ".json.zst", ".json.zstd")
SUPPORTED_EXT = PARQUET_EXT + JSON_EXT
JSON_BATCH = 10_000      # JSON lines per batch (a parquet file is read one row group at a time)


def log(msg: str, err: bool = False) -> None:
    """Timestamped line to stdout (stderr for errors) and, once the output folder exists, to <out>/filter.log."""
    global _LOGF
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, file=sys.stderr if err else sys.stdout, flush=True)
    if _A is not None and getattr(_A, "log_file", None):
        if _LOGF is None:
            _LOGF = open(_A.log_file, "a", encoding="utf-8", buffering=1)
        _LOGF.write(line + "\n")


def _init(a, counters=None):
    global _A, _CNT, _LOGF
    _A, _CNT, _LOGF = a, counters, None
    if a.nice:
        try:
            os.nice(a.nice)
        except OSError:
            pass


def _ext(path: str) -> str | None:
    name = os.path.basename(path).lower()
    for e in sorted(SUPPORTED_EXT, key=len, reverse=True):
        if name.endswith(e):
            return e
    return None


def _any_ext(name: str) -> str:
    parts = name.lower().split(".")
    if len(parts) > 2 and parts[-1] in ("gz", "zst", "zstd", "bz2", "xz", "lz4"):
        return "." + ".".join(parts[-2:])
    return "." + parts[-1] if len(parts) > 1 else "(no extension)"


def _list_inputs(root: str) -> list[str]:
    """Supported data files under `root` (symlinks followed, as in a Hugging Face cache). If there are none, stop
    with a message that lists which file types ARE there, so a wrong path or format is obvious."""
    if os.path.isfile(root):
        if _ext(root) is None:
            raise SystemExit(f"unsupported input file {root}; supported: {', '.join(SUPPORTED_EXT)}")
        return [root]
    if not os.path.isdir(root):
        raise SystemExit(f"input not found: {root}")
    found, other, example = [], Counter(), {}
    for dp, dirs, fs in os.walk(root, followlinks=True):
        dirs[:] = [d for d in dirs if not d.startswith(".")]      # skip .cache / .git / .huggingface
        for f in fs:
            if f.startswith("."):
                continue
            if _ext(f):
                found.append(os.path.join(dp, f))
            else:
                e = _any_ext(f); other[e] += 1; example.setdefault(e, os.path.join(dp, f))
    pq_files = [f for f in found if _ext(f) in PARQUET_EXT]
    if pq_files and len(pq_files) < len(found):
        # a parquet dataset folder: JSON files there are metadata (dataset_info.json, ...), not data
        log(f"  {root}: using the {len(pq_files):,} parquet files; ignoring {len(found) - len(pq_files):,} JSON file(s), "
            f"e.g. {next(f for f in found if _ext(f) not in PARQUET_EXT)}")
        found = pq_files
    if not found:
        msg = [f"no supported data files under {root} (looked for {', '.join(SUPPORTED_EXT)})"]
        if other:
            msg.append("files that ARE there, by extension: " + ", ".join(f"{e} {n:,}" for e, n in other.most_common(10)))
            msg += [f"  e.g. {example[e]}" for e, _ in other.most_common(3)]
        else:
            msg.append("the folder and its sub-folders contain no files at all")
        raise SystemExit("\n".join(msg))
    return sorted(found)


def _open_text(f: str):
    import io
    e = _ext(f)
    if e.endswith(".gz"):
        import gzip
        return gzip.open(f, "rt", encoding="utf-8", errors="replace")
    if e.endswith((".zst", ".zstd")):
        try:
            import zstandard
        except ImportError:
            raise RuntimeError(f"{f}: reading .zst files needs the zstandard package (pip install zstandard)") from None
        return io.TextIOWrapper(zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(open(f, "rb")),
                                encoding="utf-8", errors="replace")
    return open(f, "r", encoding="utf-8", errors="replace")


def _batches(f: str, a, columns=None, limit=None):
    """Yield pyarrow tables. Parquet: one row group at a time (all columns, or `columns`). JSON lines: batches of
    JSON_BATCH records as columns text / id / token_count, plus `raw` (the original line) in filter mode."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if _ext(f) in PARQUET_EXT:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        if a.text_column not in names:
            raise RuntimeError(f"{f}: no column {a.text_column!r}; columns are {names} (set --text-column)")
        n = 0
        for rg in range(pf.num_row_groups):
            cols = [c for c in columns if c in names] if columns else None
            tbl = pf.read_row_group(rg, columns=cols)
            if limit and n + tbl.num_rows > limit:
                tbl = tbl.slice(0, limit - n)
            n += tbl.num_rows
            yield tbl
            if limit and n >= limit:
                return
        return
    keep_raw = a.mode == "filter"
    recs, raws, n, bad, first = [], [], 0, 0, True
    with _open_text(f) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if first:
                first = False
                if s.startswith("["):
                    raise RuntimeError(f"{f}: this is one JSON array, not JSON lines (one object per line)")
            try:
                r = json.loads(s)
            except ValueError:
                bad += 1
                if bad > 1000 and bad > n:
                    raise RuntimeError(f"{f}: more than 1,000 lines are not valid JSON; is this really a JSON-lines file?")
                continue
            if not isinstance(r, dict):
                continue
            if n == 0 and a.text_column not in r:
                raise RuntimeError(f"{f}: no field {a.text_column!r}; fields are {sorted(r)} (set --text-column)")
            recs.append(r); n += 1
            if keep_raw:
                raws.append(s)
            if len(recs) >= JSON_BATCH or (limit and n >= limit):
                yield _json_table(recs, raws, a)
                recs, raws = [], []
                if limit and n >= limit:
                    return
    if recs:
        yield _json_table(recs, raws, a)


def _json_table(recs, raws, a):
    import pyarrow as pa
    cols = {a.text_column: [r.get(a.text_column) if isinstance(r.get(a.text_column), str) else None for r in recs]}
    if any(a.id_column in r for r in recs):
        cols[a.id_column] = [None if r.get(a.id_column) is None else str(r.get(a.id_column)) for r in recs]
    if any(isinstance(r.get("token_count"), int) for r in recs):
        cols["token_count"] = [r.get("token_count") if isinstance(r.get("token_count"), int) else None for r in recs]
    if raws:
        cols["raw"] = raws
    return pa.table(cols)


def _rel(root: str, f: str) -> str:
    """Output path inside <out>: <input folder name>/<path inside it>, so several inputs (e.g. two Nemotron subset
    folders whose files share names) never collide; the summary groups by the folder that holds each file."""
    if os.path.isdir(root):
        return os.path.join(os.path.basename(os.path.normpath(root)), os.path.relpath(f, root))
    return os.path.join(os.path.basename(os.path.dirname(os.path.abspath(f))), os.path.basename(f))


def _out_name(rel: str) -> str:
    e = _ext(rel)
    return rel if e in PARQUET_EXT else rel[: -len(e)] + ".parquet"


def _need(a):
    """Pre-filter thresholds: only in filter mode (score mode writes exact tech_* values for every row)."""
    if a.mode != "filter" or getattr(a, "no_prefilter", False):
        return None
    return (a.min_families, a.min_anchors)


def _process(job):
    """Pool entry point: any error (including SystemExit) is turned into a RuntimeError that stops the whole run
    with the file name, instead of silently killing a worker."""
    try:
        return _process_file(job)
    except BaseException as e:
        import traceback
        msg = f"FAILED {job[1]}: {type(e).__name__}: {e}"
        log(msg, err=True)
        raise RuntimeError(msg + "\n" + traceback.format_exc()) from None


def _process_file(job):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root, f = job
    a = _A
    rel = _rel(root, f)
    subset = os.path.dirname(rel)
    out_f = os.path.join(a.out, _out_name(rel))
    stat_f = os.path.join(a.out, "_stats", rel + ".json")
    if os.path.exists(stat_f):
        with open(stat_f) as fh:
            st = json.load(fh)
        st["skipped"] = True
        return st
    is_pq = _ext(f) in PARQUET_EXT
    if is_pq:
        md = pq.read_metadata(f)
        n_rows, n_parts, what = md.num_rows, md.num_row_groups, "row groups"
    else:
        n_rows, n_parts, what = None, None, "JSON lines"
    t_start = last_log = time.time()
    if _CNT is not None:
        with _CNT["files_started"].get_lock():
            _CNT["files_started"].value += 1
    log(f"start {rel}: " + (f"{n_rows:,} rows, {n_parts} row groups, " if is_pq else "JSON lines, ")
        + f"{os.path.getsize(f) / 2**20:,.0f} MB (pid {os.getpid()})")
    rng = random.Random(hash(rel) & 0xFFFFFFFF)
    st = {"file": rel, "subset": subset, "docs": 0, "kept": 0, "tokens": 0, "tokens_kept": 0, "token_source": None,
          "reasons": Counter(), "fam_hist": Counter(), "fam_freq": Counter(), "examples": []}
    out_tables = []
    need = _need(a)
    read_cols = None if a.mode == "filter" else [a.id_column, a.text_column, "token_count"]
    for b, tbl in enumerate(_batches(f, a, columns=read_cols)):
        texts = tbl.column(a.text_column).to_pylist()
        tc = tbl.column("token_count").to_pylist() if "token_count" in tbl.column_names else None
        st["token_source"] = "token_count" if tc is not None else "chars/4.6"
        cols = {k: [] for k in ("tech_keep", "tech_families", "tech_anchors", "tech_hits", "tech_density", "tech_news", "tech_depth", "tech_family_list", "tech_reason")}
        keep_idx = []
        pend_docs = pend_kept = 0
        for i, t in enumerate(texts):
            if _CNT is not None and pend_docs >= 2000:      # keep the heartbeat moving inside large row groups
                with _CNT["docs"].get_lock():
                    _CNT["docs"].value += pend_docs
                with _CNT["kept"].get_lock():
                    _CNT["kept"].value += pend_kept
                pend_docs = pend_kept = 0
            pend_docs += 1
            t = t or ""
            r = analyze(t, a.hit_cap, a.max_chars, need=need)
            keep, why = decide(r, a)
            ntok = (tc[i] if tc is not None and tc[i] is not None else len(t) / 4.6)
            st["docs"] += 1; st["tokens"] += ntok; st["reasons"][why] += 1
            if r.get("prefilter"):
                st["prefiltered"] = st.get("prefiltered", 0) + 1
            else:
                st["fam_hist"][min(r["families"], 12)] += 1
            if keep:
                st["kept"] += 1; st["tokens_kept"] += ntok; keep_idx.append(i); pend_kept += 1
                st["fam_freq"].update(r["family_list"])
            if a.examples and not r.get("prefilter") and (keep or r["families"] in (2, 3)) and rng.random() < a.example_rate:
                st["examples"].append({"file": rel, "row": st["docs"] - 1, "keep": keep, "reason": why,
                                       **{k: r[k] for k in ("families", "anchors", "hits", "density", "news", "depth", "family_list", "words")},
                                       "head": t[:1500]})
            for k, v in (("tech_keep", keep), ("tech_families", r["families"]), ("tech_anchors", r["anchors"]), ("tech_hits", r["hits"]),
                         ("tech_density", r["density"]), ("tech_news", r["news"]), ("tech_depth", r["depth"]),
                         ("tech_family_list", ",".join(r["family_list"])), ("tech_reason", why)):
                cols[k].append(v)
        if _CNT is not None:
            with _CNT["docs"].get_lock():
                _CNT["docs"].value += pend_docs
            with _CNT["kept"].get_lock():
                _CNT["kept"].value += pend_kept
        if a.mode == "filter":
            sub = tbl.take(pa.array(keep_idx, type=pa.int64()))
            for k in cols:
                if k != "tech_keep":
                    sub = sub.append_column(k, pa.array([cols[k][i] for i in keep_idx]))
            out_tables.append(sub)
        else:
            base = tbl.select([c for c in (a.id_column,) if c in tbl.column_names])
            for k, v in cols.items():
                base = base.append_column(k, pa.array(v))
            out_tables.append(base)
        now = time.time()
        if a.file_log_every and now - last_log >= a.file_log_every:
            last_log = now
            where = f"row group {b + 1}/{n_parts}, {st['docs']:,}/{n_rows:,} rows ({100 * st['docs'] / max(1, n_rows):.0f} %)" if is_pq \
                else f"{st['docs']:,} lines"
            log(f"  {rel}: {where}, kept {st['kept']:,}, {st['docs'] / (now - t_start):,.0f} docs/s")
    os.makedirs(os.path.dirname(out_f), exist_ok=True)
    os.makedirs(os.path.dirname(stat_f), exist_ok=True)
    if out_tables:
        tmp = out_f + ".tmp"
        pq.write_table(pa.concat_tables(out_tables, promote_options="default"), tmp, compression="zstd")
        os.replace(tmp, out_f)
    if st["docs"] == 0:
        log(f"WARNING {rel}: the file has no documents", err=True)
    st["reasons"] = dict(st["reasons"]); st["fam_hist"] = {str(k): v for k, v in st["fam_hist"].items()}; st["fam_freq"] = dict(st["fam_freq"])
    with open(stat_f + ".tmp", "w") as fh:
        json.dump(st, fh)
    os.replace(stat_f + ".tmp", stat_f)
    el = time.time() - t_start
    log(f"done  {rel}: kept {st['kept']:,}/{st['docs']:,} docs ({100 * st['kept'] / max(1, st['docs']):.1f} %), "
        f"{st['tokens_kept'] / 1e6:,.1f} M tokens kept, {el / 60:.1f} min ({st['docs'] / max(el, 1e-9):,.0f} docs/s)")
    if _CNT is not None:
        with _CNT["files_done"].get_lock():
            _CNT["files_done"].value += 1
    return st



def _summarize(stats, a, elapsed):
    tot = Counter(); by_sub = {}
    reasons = Counter(); hist = Counter(); freq = Counter()
    for s in stats:
        b = by_sub.setdefault(s["subset"], Counter())
        for k in ("docs", "kept", "tokens", "tokens_kept"):
            b[k] += s[k]; tot[k] += s[k]
        reasons.update(s["reasons"]); hist.update({int(k): v for k, v in s["fam_hist"].items()}); freq.update(s["fam_freq"])
        tot["prefiltered"] += s.get("prefiltered", 0)
    rule = (f"families >= {a.min_families}, anchors >= {a.min_anchors}, density >= {a.min_density}/1k words, news <= {a.max_news_ratio} x hits, "
            f"words >= {a.min_words}, depth >= {a.min_depth}")
    summary = {"inputs": a.inputs, "mode": a.mode, "rule": rule, "files": len(stats), "seconds": round(elapsed),
               "docs": tot["docs"], "kept": tot["kept"], "tokens": round(tot["tokens"]), "tokens_kept": round(tot["tokens_kept"]),
               "token_source": sorted({s["token_source"] for s in stats if s["token_source"]}),
               "reasons": dict(reasons), "family_count_hist": {str(k): hist[k] for k in sorted(hist)},
               "kept_family_frequency": dict(freq.most_common()), "by_subset": {k: dict(v) for k, v in sorted(by_sub.items())}}
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    pct = lambda x, y: f"{100 * x / y:.1f} %" if y else "-"
    L = [f"# Technical-ML filter: {', '.join(a.inputs)}", "",
         f"- rule: {rule}", f"- mode: {a.mode}; {len(stats)} files; {round(elapsed / 60, 1)} min; tokens from {', '.join(summary['token_source'])}",
         f"- kept **{tot['kept']:,} / {tot['docs']:,} docs ({pct(tot['kept'], tot['docs'])})**, "
         f"**{tot['tokens_kept'] / 1e6:,.1f} M / {tot['tokens'] / 1e6:,.1f} M tokens ({pct(tot['tokens_kept'], tot['tokens'])})**", "",
         "| subset | docs | kept | kept % | M tokens | M tokens kept | tokens kept % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in sorted(by_sub.items()):
        L.append(f"| {k} | {v['docs']:,} | {v['kept']:,} | {pct(v['kept'], v['docs'])} | {v['tokens'] / 1e6:,.1f} | {v['tokens_kept'] / 1e6:,.1f} | {pct(v['tokens_kept'], v['tokens'])} |")
    L += ["", "| reject reason (first failing rule) | docs | share |", "|---|---:|---:|"]
    L += [f"| {k} | {v:,} | {pct(v, tot['docs'])} |" for k, v in reasons.most_common()]
    L += ["", f"Pre-filtered (too few candidate families to ever pass, regexes skipped; not in the histogram below): "
              f"{tot['prefiltered']:,} docs ({pct(tot['prefiltered'], tot['docs'])}). Use --no-prefilter for an exact histogram.",
          "", "| distinct families in doc | docs | share |", "|---:|---:|---:|"]
    L += [f"| {k}{'+' if k == 12 else ''} | {hist[k]:,} | {pct(hist[k], tot['docs'])} |" for k in sorted(hist)]
    L += ["", "Most frequent families in kept docs: " + ", ".join(f"{k} {v:,}" for k, v in freq.most_common(15))]
    with open(os.path.join(a.out, "summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    log("summary written to " + os.path.join(a.out, "summary.md") + "\n" + "\n".join(L))


def _footer_rows(f):
    import pyarrow.parquet as pq
    try:
        return pq.read_metadata(f).num_rows
    except Exception:      # broken / footer-less file: reported, counted as 0
        return -1


def _discover(a) -> list:
    jobs = []
    for root in a.inputs:
        log(f"listing {root} ...")
        fs = _list_inputs(root)
        fs = fs[: a.limit_files] if a.limit_files else fs
        kinds = Counter(_ext(f) for f in fs)
        gb = sum(os.path.getsize(f) for f in fs) / 2**30
        log(f"  {len(fs):,} files ({', '.join(f'{k} {v:,}' for k, v in kinds.most_common())}), {gb:,.1f} GB; first: {fs[0]}")
        jobs += [(root, f) for f in fs]
    return jobs


def check(a) -> None:
    """--check: list inputs, show the format and fields of the first file of each input, and score its first
    --check-docs documents; writes nothing."""
    log(f"keyword engine: {ENGINE}")
    jobs = _discover(a)
    seen = set()
    for root, f in jobs:
        if root in seen:
            continue
        seen.add(root)
        log(f"check {f}")
        if _ext(f) in PARQUET_EXT:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(f)
            log(f"  parquet: {pf.metadata.num_rows:,} rows, {pf.num_row_groups} row groups, columns {pf.schema_arrow.names}")
        n = kept = 0; hist = Counter(); why = Counter(); first = None
        for tbl in _batches(f, a, columns=[a.text_column, "token_count", a.id_column], limit=a.check_docs):
            for t in tbl.column(a.text_column).to_pylist():
                t = t or ""
                first = first or t
                r = analyze(t, a.hit_cap, a.max_chars, need=None if a.no_prefilter else (a.min_families, a.min_anchors)); k, w = decide(r, a)
                n += 1; kept += k; why[w] += 1
                hist["pre-filtered" if r.get("prefilter") else min(r["families"], 6)] += 1
        log(f"  first document: {first[:200]!r}" if first else "  no documents read")
        log(f"  first {n:,} docs: kept {kept:,} ({100 * kept / max(1, n):.1f} %) | families per doc "
            + ", ".join(f"{k}{'+' if k == 6 else ''}: {v:,}" for k, v in sorted(hist.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])))
            + f" | reasons {dict(why.most_common())}")
    log("check finished: nothing was written. Run again without --check to filter.")


def run(a):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp
    global _A
    a.log_file = None
    _A = a
    log(f"filter_technical: inputs {a.inputs} -> {a.out} | mode {a.mode}, {a.workers} workers, nice {a.nice} | "
        f"rule families>={a.min_families} anchors>={a.min_anchors} density>={a.min_density} news<={a.max_news_ratio}x words>={a.min_words}")
    log(f"keyword engine: {ENGINE}; pre-filter {'on' if _need(a) else 'off'}")
    jobs = _discover(a)                       # stops with a clear message before anything is created
    os.makedirs(a.out, exist_ok=True)
    a.log_file = os.path.join(a.out, "filter.log")
    log(f"log file {a.log_file}; {len(jobs):,} files to process")
    total_rows = None
    if any(_ext(f) not in PARQUET_EXT for _, f in jobs):
        log("JSON inputs: their row count is unknown in advance, so the ETA is estimated from finished files")
    elif not a.no_count:
        log(f"reading {len(jobs):,} parquet footers for the total row count (skip with --no-count) ...")
        with ThreadPoolExecutor(16) as ex:
            rows = list(ex.map(_footer_rows, [f for _, f in jobs]))
        bad = [f for (_, f), n in zip(jobs, rows) if n < 0]
        total_rows = sum(n for n in rows if n > 0)
        log(f"  {total_rows:,} rows in total" + (f"; {len(bad)} unreadable footers, e.g. {bad[:3]}" if bad else ""))
    done_before = sum(os.path.exists(os.path.join(a.out, "_stats", _rel(r, f) + ".json")) for r, f in jobs)
    if done_before:
        log(f"resume: {done_before:,} of {len(jobs):,} files already done and will be skipped")
    counters = {k: mp.Value("q", 0) for k in ("docs", "kept", "files_started", "files_done")}
    t0 = time.time(); stats = []
    stop = threading.Event()

    def heartbeat():
        last_t, last_docs = t0, 0
        while not stop.wait(a.log_every):
            now, docs, kept = time.time(), counters["docs"].value, counters["kept"].value
            rate = (docs - last_docs) / max(now - last_t, 1e-9); avg = docs / max(now - t0, 1e-9)
            last_t, last_docs = now, docs
            fdone = counters["files_done"].value
            busy = counters["files_started"].value - fdone
            msg = (f"heartbeat: files done {len(stats):,}/{len(jobs):,} ({busy} in progress) | docs {docs:,}"
                   + (f"/{total_rows:,} ({100 * docs / max(1, total_rows):.2f} %)" if total_rows else "")
                   + f" | kept {kept:,} ({100 * kept / max(1, docs):.2f} %) | {rate:,.0f} docs/s now, {avg:,.0f} avg")
            eta = None
            if total_rows and avg > 0:
                eta = (total_rows - docs) / avg
            elif fdone:
                eta = (now - t0) / fdone * (len(jobs) - done_before - fdone)
            if eta is not None:
                msg += f" | ETA {eta / 60:,.0f} min" if eta < 7200 else f" | ETA {eta / 3600:,.1f} h ({eta / 86400:,.1f} days)"
            log(msg)
    hb = threading.Thread(target=heartbeat, daemon=True); hb.start()
    try:
        with Pool(a.workers, initializer=_init, initargs=(a, counters)) as pool:
            for st in pool.imap_unordered(_process, jobs):
                stats.append(st)
                if st.get("skipped") and len(stats) % 100 == 0:
                    log(f"  ... {len(stats):,} files skipped/done so far")
    except Exception as e:
        stop.set()
        log(f"RUN STOPPED: {str(e).splitlines()[0]}", err=True)
        log("finished files are kept; fix the problem and rerun the same command to resume", err=True)
        sys.exit(1)
    finally:
        stop.set()
    log(f"all {len(jobs):,} files finished in {(time.time() - t0) / 3600:.2f} h")
    if a.examples:
        ex = [e for s in stats for e in s.get("examples", [])]
        random.Random(0).shuffle(ex)
        kept = [e for e in ex if e["keep"]][: a.examples]
        near = [e for e in ex if not e["keep"]][: a.examples]
        with open(os.path.join(a.out, "examples.jsonl"), "w", encoding="utf-8") as fh:
            for e in kept + near:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        log(f"wrote {len(kept)} kept + {len(near)} rejected near-miss examples to {os.path.join(a.out, 'examples.jsonl')}")
    _summarize(stats, a, time.time() - t0)
    if sum(s["docs"] for s in stats) == 0:
        log("WARNING: no documents were read from any input file", err=True)
        sys.exit(1)



# --------------------------------------------------------------------------------------------------------------- #
# Self-test / explain
# --------------------------------------------------------------------------------------------------------------- #
SELFTEST = [
    (True, "technical", "To train the network we minimise the cross-entropy loss with stochastic gradient descent. "
     "The gradients are computed by backpropagation through every hidden layer, and the softmax output layer turns the logits into "
     "probabilities. A learning rate of 0.001 with weight decay reduces overfitting; we pick the hyperparameters on a validation set. " * 4),
    (False, "ai_news", "OpenAI's CEO announced a new partnership today. The start-up raised $6 billion from investors to train "
     "large language models, and customers can sign up for the new enterprise pricing. Analysts say generative AI and machine "
     "learning will transform the market for artificial intelligence, while regulators debate deepfakes and AI safety. " * 4),
    (False, "physics", "The entropy of the gas increases. The Hamiltonian's eigenvalues give the energy levels, and the "
     "partition function follows from a Monte Carlo simulation of the Markov chain. The tensor of inertia has a Jacobian determinant "
     "and the momentum is conserved in this vector space. " * 4),
    (False, "one_keyword", "Neural networks are everywhere. This blog post is about how a classifier changed our favourite "
     "app and why people love it. Nothing else here is technical at all, just stories about users, design and colours. " * 6),
]


def selftest(a) -> int:
    bad = 0
    for want, name, text in SELFTEST:
        r = analyze(text); keep, why = decide(r, a)
        ok = keep == want; bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {name:12s} keep={keep!s:5s} ({why}) families={r['families']} anchors={r['anchors']} "
              f"density={r['density']} news={r['news']} depth={r['depth_list']} {r['family_list']}")
    print(f"{len(ANCHORS)} anchor families, {len(AMBIGUOUS)} ambiguous families")
    return bad


def explain(path: str, a) -> None:
    text = sys.stdin.read() if path == "-" else open(path, encoding="utf-8", errors="ignore").read()
    r = analyze(text, a.hit_cap, a.max_chars); keep, why = decide(r, a)
    print(json.dumps({"keep": keep, "reason": why, **{k: v for k, v in r.items()}}, indent=1, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="parquet file(s) or tree(s), then the output folder")
    ap.add_argument("--mode", choices=["filter", "score"], default="filter",
                    help="filter: write passing rows with all columns; score: write id + tech_* for every row")
    ap.add_argument("--min-families", type=int, default=3, help="distinct concept families required (>= 2; default 3)")
    ap.add_argument("--min-anchors", type=int, default=1, help="unambiguous-ML families required (default 1)")
    ap.add_argument("--min-density", type=float, default=2.0, help="ML keyword hits per 1,000 words (default 2.0)")
    ap.add_argument("--max-news-ratio", type=float, default=0.5, help="reject if news/business hits > this x ML hits (default 0.5)")
    ap.add_argument("--min-depth", type=int, default=0, help="distinct depth-marker types required (latex, equation, code, ...; default 0)")
    ap.add_argument("--min-words", type=int, default=100)
    ap.add_argument("--hit-cap", type=int, default=10, help="max hits counted per family per document (default 10)")
    ap.add_argument("--max-chars", type=int, default=300_000, help="scan at most this many characters (0 = all)")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--id-column", default="id")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--nice", type=int, default=10, help="niceness of the workers (default 10, so a training job keeps priority)")
    ap.add_argument("--limit-files", type=int, default=None, help="only the first N files of each input (smoke tests)")
    ap.add_argument("--log-every", type=int, default=60, help="seconds between overall heartbeat lines (default 60)")
    ap.add_argument("--file-log-every", type=int, default=300, help="seconds between per-file progress lines of each worker (default 300; 0 = off)")
    ap.add_argument("--no-count", action="store_true", help="skip reading all parquet footers for the total row count / ETA")
    ap.add_argument("--examples", type=int, default=0, help="write N kept + N rejected near-miss docs to examples.jsonl")
    ap.add_argument("--example-rate", type=float, default=0.01, help="sampling rate for --examples (default 0.01)")
    ap.add_argument("--explain", metavar="FILE", help="score one text file ('-' = stdin) and print the breakdown")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-prefilter", action="store_true",
                    help="run every family regex on every doc (exact family histogram in summary.md; slower; kept docs are identical)")
    ap.add_argument("--check", action="store_true", help="list the inputs and score the first --check-docs docs of each; write nothing")
    ap.add_argument("--check-docs", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.min_families < 2:
        ap.error("--min-families must be >= 2: a single keyword family never qualifies a document")
    if a.selftest:
        sys.exit(1 if selftest(a) else 0)
    if a.explain:
        explain(a.explain, a); return
    if a.check:
        if not a.inputs:
            ap.error("give at least one input")
        a.out, a.log_file = None, None
        global _A
        _A = a
        check(a); return
    if len(a.inputs) < 2:
        ap.error("give at least one input and an output folder")
    a.out = os.path.abspath(a.inputs[-1]); a.inputs = a.inputs[:-1]
    run(a)


if __name__ == "__main__":
    main()
