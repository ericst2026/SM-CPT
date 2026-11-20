#!/usr/bin/env python3
"""Keyword filter: pull AI / ML documents out of web-scale corpora (FineWeb-Edu, Nemotron-CC, ...) for
continued pre-training. Self-contained - copy this one file to the data server.

    pip install pyahocorasick pyarrow zstandard          # orjson optional (faster JSON)

    # 1) calibrate on a sample: score distribution, accept rate, accepted / near-miss examples
    python filter_ai_corpus.py --input '/data/fineweb-edu/**/*.parquet' --dry-run 50000

    # 2) full run (resumable: finished input files are skipped)
    python filter_ai_corpus.py --out /data/ai_cpt --workers 64 \
        --input '/data/fineweb-edu/**/*.parquet' \
        --input '/data/nemotron-cc/**/*.jsonl.zstd'

    # several servers: same command with --shard 0/4, 1/4, 2/4, 3/4

How it works
  * Two Aho-Corasick automata (pyahocorasick): ~700 lower-case phrase variants scanned on the lower-cased
    text, plus case-sensitive acronyms (AI, LLM, GPT, CNN, ...) scanned on the original text. One linear
    pass per document, independent of the lexicon size.
  * Word boundaries are checked on every hit (a trailing "s"/"es" plural is allowed), so "ai" never
    fires inside "said" and "gan" not inside "organ". Space-separated phrases also match with a hyphen.
  * score = sum over distinct terms of weight x (1 + 0.25 per repeat, max 5), minus negative contexts
    (artificial insemination, CNN news, power transformers, ...).
    weight 3 = unambiguous ("backpropagation"), 2 = strong ("neural network"), 1 = weak ("dataset").
  * accept if  score >= --min-score  AND  distinct strong (w >= 2) terms >= --min-strong
               AND  density (score per 1000 words) >= --min-density
    The density test drops long pages that mention AI once in passing.

Output (--out)
  <input-root-name>/<same relative path>.parquet   accepted docs: text + metadata columns that exist in the
                                                   input (id, url, score, int_score, dump, warc_record_id,
                                                   label, ...) + ai_score, ai_density, ai_topic, ai_hits
  _general/<input-root-name>/<path>.parquet        random sample of NON-AI docs (--general-rate) for replay,
                                                   so the CPT mix can keep general text against forgetting
  _stats/<input-root-name>/<path>.json             per-file counts; its presence marks the file as done
  _summary.json                                    totals over all finished files (docs, chars, ~tokens, topics)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import os
import random
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import ahocorasick
except ImportError:
    sys.exit("pip install pyahocorasick")

try:
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads

# --------------------------------------------------------------------------------------------------- #
# Lexicon: (weight, topic, "variant | variant | ...")  - lower-case, matched with word boundaries.
# A space inside a phrase also matches a hyphen ("fine tuning" -> "fine-tuning"). Plurals come free.
# Keep out words that are common outside AI ("model", "learning", "data", "agent"): they only cost time.
# --------------------------------------------------------------------------------------------------- #
LEXICON = [
    # --- field names -------------------------------------------------------------------------------
    (3, "core", "artificial intelligence | machine learning | deep learning | reinforcement learning"),
    (3, "core", "supervised learning | unsupervised learning | semi supervised learning | self supervised learning"),
    (3, "core", "natural language processing | computer vision | generative ai | statistical learning"),
    (2, "core", "data science | data mining | pattern recognition | machine intelligence | cognitive computing"),
    (3, "core", "artificial general intelligence | narrow ai | general ai | superintelligence | strong ai"),
    # --- neural networks / deep learning -----------------------------------------------------------
    (3, "dl", "neural network | neural net | artificial neuron | perceptron | multilayer perceptron | multi layer perceptron"),
    (3, "dl", "convolutional neural network | convolutional layer | recurrent neural network | long short term memory"),
    (3, "dl", "backpropagation | back propagation | gradient descent | stochastic gradient descent | vanishing gradient | exploding gradient"),
    (3, "dl", "activation function | loss function | cross entropy loss | softmax | relu | leaky relu | gelu | sigmoid activation"),
    (3, "dl", "dropout layer | batch normalization | layer normalization | residual connection | skip connection | weight decay"),
    (3, "dl", "hidden layer | fully connected layer | pooling layer | max pooling | embedding layer | feed forward network | feedforward network"),
    (3, "dl", "learning rate | hyperparameter | hyperparameters | hyperparameter tuning | early stopping | mini batch | minibatch"),
    (3, "dl", "adam optimizer | adamw | rmsprop | adagrad | learning rate schedule | warmup steps | cosine annealing"),
    (3, "dl", "autoencoder | variational autoencoder | generative adversarial network | diffusion model | denoising diffusion | latent diffusion"),
    (3, "dl", "normalizing flow | energy based model | score matching | flow matching | autoregressive model"),
    (3, "dl", "graph neural network | graph convolutional network | message passing neural network | spiking neural network"),
    (3, "dl", "transfer learning | fine tuning | finetuning | pretraining | pre training | pretrained model | pre trained model"),
    (3, "dl", "knowledge distillation | model distillation | model quantization | model pruning | mixed precision training"),
    (2, "dl", "deep neural | neural architecture | neural architecture search | weight initialization | xavier initialization"),
    # --- transformers / LLMs -----------------------------------------------------------------------
    (3, "llm", "transformer model | transformer architecture | transformer encoder | transformer decoder | vision transformer"),
    (3, "llm", "self attention | multi head attention | multihead attention | cross attention | attention mechanism | scaled dot product attention"),
    (3, "llm", "positional encoding | positional embedding | rotary position embedding | encoder decoder | sequence to sequence | seq2seq"),
    (3, "llm", "large language model | language model | foundation model | small language model | multimodal model"),
    (3, "llm", "next token prediction | tokenizer | tokenization | byte pair encoding | wordpiece | sentencepiece | context window"),
    (3, "llm", "prompt engineering | in context learning | chain of thought | few shot prompting | zero shot prompting | system prompt"),
    (3, "llm", "instruction tuning | supervised fine tuning | reinforcement learning from human feedback | direct preference optimization"),
    (3, "llm", "reward model | preference optimization | constitutional ai | low rank adaptation | parameter efficient fine tuning"),
    (3, "llm", "retrieval augmented generation | vector database | vector store | semantic search | sentence embedding | text embedding"),
    (3, "llm", "kv cache | flash attention | speculative decoding | beam search | greedy decoding | top p sampling | nucleus sampling"),
    (3, "llm", "mixture of experts | scaling laws | emergent abilities | hallucination in llms | llm hallucination | prompt injection"),
    (3, "llm", "ai agent | llm agent | agentic | tool calling | function calling | multi agent system"),
    (3, "llm", "chatgpt | gpt 3 | gpt 4 | gpt 4o | openai | anthropic claude | google gemini | llama 2 | llama 3 | mistral 7b | deepseek"),
    (2, "llm", "chatbot | chat bot | conversational ai | virtual assistant | text generation | image generation | text to image"),
    # --- classical ML ------------------------------------------------------------------------------
    (3, "ml", "random forest | decision tree classifier | gradient boosting | xgboost | lightgbm | catboost | adaboost"),
    (3, "ml", "support vector machine | kernel trick | k nearest neighbors | k nearest neighbours | naive bayes classifier | naive bayes"),
    (3, "ml", "logistic regression | linear regression model | ridge regression | lasso regression | polynomial regression"),
    (3, "ml", "k means clustering | k means | hierarchical clustering | dbscan | gaussian mixture model | expectation maximization"),
    (3, "ml", "principal component analysis | dimensionality reduction | t sne | umap embedding | feature extraction | feature engineering"),
    (3, "ml", "overfitting | underfitting | bias variance tradeoff | bias variance trade off | regularization technique | cross validation"),
    (3, "ml", "training set | test set | validation set | training data | labeled data | labelled data | unlabeled data | data augmentation"),
    (3, "ml", "confusion matrix | precision and recall | f1 score | roc curve | auc score | mean squared error | classification accuracy"),
    (3, "ml", "ensemble learning | ensemble methods | bagging and boosting | active learning | online learning algorithm"),
    (3, "ml", "few shot learning | zero shot learning | meta learning | continual learning | catastrophic forgetting | contrastive learning"),
    (3, "ml", "representation learning | metric learning | federated learning | multi task learning | curriculum learning | imitation learning"),
    (3, "ml", "collaborative filtering | recommender system | recommendation system | recommendation engine | anomaly detection"),
    (3, "ml", "bayesian network | hidden markov model | markov random field | conditional random field | variational inference | gaussian process"),
    (2, "ml", "classifier | classification model | regression model | clustering algorithm | supervised model | predictive model | predictive modeling"),
    (2, "ml", "feature vector | feature importance | one hot encoding | class imbalance | imbalanced dataset | training loop | training epoch"),
    (1, "ml", "dataset | datasets | epoch | epochs | inference time | ground truth labels | benchmark dataset | kaggle"),
    # --- NLP / vision / speech ---------------------------------------------------------------------
    (3, "nlp", "word embedding | word2vec | glove embeddings | bag of words | tf idf | n gram language model | named entity recognition"),
    (3, "nlp", "sentiment analysis | machine translation | text classification | text summarization | question answering system"),
    (3, "nlp", "part of speech tagging | dependency parsing | coreference resolution | topic modeling | latent dirichlet allocation"),
    (3, "cv", "image classification | object detection | image segmentation | semantic segmentation | instance segmentation"),
    (3, "cv", "facial recognition | face recognition | image recognition | pose estimation | optical character recognition | yolo detector"),
    (3, "cv", "convolution kernel | convolutional filter | feature map | bounding box regression | resnet | alexnet | imagenet"),
    (3, "speech", "speech recognition | automatic speech recognition | speech to text | text to speech | speech synthesis | voice cloning"),
    # --- RL / search / planning / classical AI -----------------------------------------------------
    (3, "rl", "markov decision process | q learning | deep q network | policy gradient | actor critic | proximal policy optimization"),
    (3, "rl", "bellman equation | value iteration | policy iteration | temporal difference | reward function | reward shaping"),
    (3, "rl", "exploration exploitation | epsilon greedy | multi armed bandit | monte carlo tree search | self play | alphago | alphazero"),
    (3, "symbolic", "expert system | knowledge representation | knowledge graph | inference engine | forward chaining | backward chaining"),
    (3, "symbolic", "a* search | heuristic search | minimax algorithm | alpha beta pruning | constraint satisfaction problem | automated planning"),
    (3, "symbolic", "genetic algorithm | evolutionary algorithm | swarm intelligence | particle swarm optimization | fuzzy logic | neuro symbolic"),
    (3, "symbolic", "intelligent agent | rational agent | symbolic ai | connectionism | turing test | chinese room"),
    # --- robotics / autonomy -----------------------------------------------------------------------
    (3, "robotics", "self driving car | autonomous vehicle | autonomous driving | robot learning | motion planning | path planning algorithm"),
    (3, "robotics", "simultaneous localization and mapping | sensor fusion | embodied ai | humanoid robot | robotic manipulation"),
    # --- MLOps / tools -----------------------------------------------------------------------------
    (3, "tools", "pytorch | tensorflow | keras | scikit learn | sklearn | hugging face | huggingface | jax framework | onnx | tensorrt"),
    (3, "tools", "mlops | model deployment | model serving | model monitoring | feature store | experiment tracking | automl"),
    (2, "tools", "jupyter notebook | google colab | weights and biases | mlflow | langchain | llamaindex | opencv | nltk | spacy"),
    (2, "tools", "gpu training | tpu | cuda kernel | ai accelerator | neural processing unit | inference engine"),
    # --- trustworthy AI / society ------------------------------------------------------------------
    (3, "ethics", "ai ethics | ai safety | ai alignment | ai regulation | ai governance | ai act | responsible ai | trustworthy ai"),
    (3, "ethics", "algorithmic bias | algorithmic fairness | fairness in machine learning | explainable ai | interpretability | explainability"),
    (3, "ethics", "adversarial example | adversarial attack | adversarial robustness | data poisoning | membership inference | differential privacy"),
    (3, "ethics", "deepfake | deep fake | ai generated | ai art | existential risk from ai | mechanistic interpretability"),
    (2, "ethics", "job automation | automation of jobs | ai and jobs | ai in healthcare | ai in education | ai in finance"),
    # --- history / people --------------------------------------------------------------------------
    (3, "history", "alan turing | john mccarthy | marvin minsky | frank rosenblatt | geoffrey hinton | yann lecun | yoshua bengio"),
    (3, "history", "dartmouth workshop | dartmouth conference | ai winter | lighthill report | deep blue | eliza chatbot | shakey the robot"),
    (3, "history", "andrew ng | fei fei li | demis hassabis | ilya sutskever | judea pearl | richard sutton | jurgen schmidhuber"),
    # --- research ----------------------------------------------------------------------------------
    (3, "research", "neurips | icml | iclr | cvpr | iccv | emnlp | aaai conference | ijcai | arxiv preprint | deepmind"),
    (1, "research", "state of the art | ablation study | benchmark results | baseline model"),
    # --- math that is AI-specific (generic math terms are left out: they would pull in all of math) ---
    (2, "math", "cross entropy | kl divergence | kullback leibler | maximum likelihood estimation | log likelihood"),
    (2, "math", "gradient of the loss | chain rule for backpropagation | jacobian matrix | hessian matrix | convex optimization"),
    (1, "math", "softmax function | sigmoid function | tanh | bayes theorem | posterior distribution | latent variable"),
    # --- code ---------------------------------------------------------------------------------------
    (3, "code", "import torch | from torch import | torch.nn | nn.module | torch.optim | .backward() | optimizer.step() | dataloader("),
    (3, "code", "import tensorflow | tf.keras | from keras | import keras | model.compile( | from sklearn | import sklearn"),
    (3, "code", "train_test_split | .fit_transform( | from transformers import | automodel | autotokenizer | import jax"),
    (3, "code", "from peft import | loraconfig | sfttrainer | import xgboost | import lightgbm | gym.make( | stable_baselines3"),
    (1, "code", "import numpy | import pandas | np.array( | pd.dataframe( | matplotlib.pyplot"),
]

# Case-sensitive acronyms / names, matched on the original text.
LEXICON_CS = [
    (2, "core", "AI | A.I. | AIs"),
    (1, "core", "ML"),
    (3, "nlp", "NLP | NLU | NLG"),
    (3, "llm", "LLM | LLMs | GPT | GPTs | RLHF | LoRA | QLoRA | PEFT | MoE | RAG pipeline"),
    (2, "llm", "BERT | RoBERTa | LLaMA | Llama | Qwen | Mistral AI | Gemini | Claude | Copilot | Midjourney | DALL-E | Stable Diffusion"),
    (3, "dl", "CNN | CNNs | RNN | RNNs | LSTM | LSTMs | GRU | GAN | GANs | VAE | VAEs | MLP | MLPs | ResNet | ViT | U-Net"),
    (3, "ml", "SVM | SVMs | KNN | PCA"),
    (3, "rl", "DQN | PPO algorithm | MDP | MCTS | DDPG"),
    (2, "core", "AGI | XAI | GenAI"),
    (1, "tools", "GPU | GPUs | TPU | TPUs | CUDA"),
]

# Negative contexts: subtract once per distinct hit.
NEGATIVE = [
    (6, "artificial insemination"), (4, "adobe illustrator"), (4, ".ai file"), (4, "cnn news"),
    (4, "cnn reported"), (4, "cnn anchor"), (4, "power transformer"), (4, "voltage transformer"),
    (4, "step up transformer"), (4, "step down transformer"), (4, "distribution transformer"),
    (5, "optimus prime"), (5, "autobots"), (5, "decepticon"), (3, "neural tube"), (3, "neural crest"),
    (4, "ai weiwei"), (3, "amnesty international"), (3, "lora radio"), (3, "lorawan"),
]

META_COLUMNS = ["id", "url", "dump", "date", "file_path", "language", "language_score", "token_count",
                "score", "int_score", "warc_record_id", "label", "quality", "source"]

# --------------------------------------------------------------------------------------------------- #
# Automaton
# --------------------------------------------------------------------------------------------------- #
_AUTO_LOW = _AUTO_CS = None
_TERMS: list[tuple[str, float, str]] = []      # term id -> (name, weight, topic); weight < 0 = negative


def _variants(phrase: str) -> set[str]:
    p = phrase.strip()
    out = {p}
    if " " in p:
        out.add(p.replace(" ", "-"))
    return out


def build_automata():
    global _AUTO_LOW, _AUTO_CS, _TERMS
    _TERMS = []
    low, cs = ahocorasick.Automaton(), ahocorasick.Automaton()

    def add(auto, key, tid):
        # (term id, length, check left boundary, check right boundary)
        auto.add_word(key, (tid, len(key), key[0].isalnum(), key[-1].isalnum()))

    for table, auto, lower in ((LEXICON, low, True), (LEXICON_CS, cs, False)):
        for w, topic, group in table:
            for phrase in group.split("|"):
                phrase = phrase.strip()
                if not phrase:
                    continue
                tid = len(_TERMS)
                _TERMS.append((phrase, float(w), topic))
                for v in _variants(phrase.lower() if lower else phrase):
                    add(auto, v, tid)
    for w, phrase in NEGATIVE:
        tid = len(_TERMS)
        _TERMS.append((phrase, -float(w), "neg"))
        add(low, phrase, tid)
    low.make_automaton()
    cs.make_automaton()
    _AUTO_LOW, _AUTO_CS = low, cs


def _scan(auto, text: str, counts: dict):
    n_text = len(text)
    for end, (tid, n, lb, rb) in auto.iter(text):
        start = end - n + 1
        if lb and start > 0 and text[start - 1].isalnum():
            continue
        if rb:
            nxt = end + 1
            if nxt < n_text and text[nxt].isalnum():
                # allow plural "s" / "es"
                if text[nxt] == "s":
                    nxt += 1
                elif text[nxt:nxt + 2] == "es":
                    nxt += 2
                else:
                    continue
                if nxt < n_text and text[nxt].isalnum():
                    continue
        counts[tid] = counts.get(tid, 0) + 1


def score_doc(text: str) -> tuple[float, int, float, dict]:
    """(score, distinct strong terms, density per 1000 words, {term id: count})"""
    counts: dict[int, int] = {}
    _scan(_AUTO_LOW, text.lower().replace("\n", " "), counts)
    _scan(_AUTO_CS, text, counts)
    if not counts:
        return 0.0, 0, 0.0, counts
    s, strong = 0.0, 0
    for tid, c in counts.items():
        w = _TERMS[tid][1]
        if w < 0:
            s += w
            continue
        s += w * (1.0 + 0.25 * (min(c, 5) - 1))
        if w >= 2:
            strong += 1
    words = text.count(" ") + 1
    return s, strong, 1000.0 * s / words, counts


def describe(counts: dict, k: int = 8) -> tuple[str, str]:
    """(top topic, 'term:count,...' of the strongest hits)"""
    by_topic: Counter = Counter()
    for tid, c in counts.items():
        name, w, topic = _TERMS[tid]
        if w > 0:
            by_topic[topic] += w * min(c, 5)
    top = sorted((tid for tid in counts if _TERMS[tid][1] > 0),
                 key=lambda t: -_TERMS[t][1] * min(counts[t], 5))[:k]
    return (by_topic.most_common(1)[0][0] if by_topic else ""), ",".join(f"{_TERMS[t][0]}:{counts[t]}" for t in top)


# --------------------------------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------------------------------- #
def _open_text(path: str):
    if path.endswith((".zst", ".zstd")):
        import zstandard
        fh = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(fh),
                                encoding="utf-8", errors="replace")
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def iter_batches(path: str, text_key: str, batch_size: int = 4096):
    """Yields (texts, meta) where meta is a pyarrow RecordBatch (parquet) or a list of dicts (jsonl)."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        if text_key not in names:
            raise KeyError(f"{path}: no column {text_key!r} (has {names})")
        cols = [text_key] + [c for c in META_COLUMNS if c in names and c != text_key]
        for b in pf.iter_batches(batch_size=batch_size, columns=cols):
            yield b.column(text_key).to_pylist(), b
    else:
        buf = []
        with _open_text(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    buf.append(_loads(line))
                except ValueError:
                    continue
                if len(buf) >= batch_size:
                    yield [r.get(text_key) or "" for r in buf], buf
                    buf = []
        if buf:
            yield [r.get(text_key) or "" for r in buf], buf


class Writer:
    """Parquet writer that opens lazily and renames into place on close."""

    def __init__(self, path: str):
        self.path, self.tmp, self.w, self.n = path, path + ".tmp", None, 0

    def write(self, table):
        import pyarrow.parquet as pq
        if table.num_rows == 0:
            return
        if self.w is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.w = pq.ParquetWriter(self.tmp, table.schema, compression="zstd")
        self.w.write_table(table)
        self.n += table.num_rows

    def close(self):
        if self.w is not None:
            self.w.close()
            os.replace(self.tmp, self.path)


def _to_table(texts, meta, idx, extra: dict, text_key: str, jsonl_keys: list[str]):
    import pyarrow as pa
    if isinstance(meta, list):                          # jsonl rows: one all-string schema per file
        cols = {text_key: pa.array([texts[i] for i in idx], pa.string())}
        for c in jsonl_keys:
            cols[c] = pa.array([None if meta[i].get(c) is None else str(meta[i][c]) for i in idx], pa.string())
    else:
        t = meta.take(pa.array(idx, pa.int64()))
        cols = {n: t.column(n) for n in t.schema.names}
    for k, (vals, typ) in extra.items():
        cols[k] = pa.array(vals, typ)
    return pa.table(cols)


# --------------------------------------------------------------------------------------------------- #
# One file
# --------------------------------------------------------------------------------------------------- #
def _init_worker():
    build_automata()


def process_file(job: dict) -> dict:
    import pyarrow as pa
    a = job["args"]
    if _AUTO_LOW is None:
        build_automata()
    rng = random.Random(zlib.crc32(job["rel"].encode()))
    out_ai, out_gen = Writer(job["out_ai"]), Writer(job["out_gen"])
    st = Counter()
    topics = Counter()
    t0 = time.time()
    jsonl_keys = None
    try:
        for texts, meta in iter_batches(job["path"], a.text_key):
            if jsonl_keys is None and isinstance(meta, list) and meta:
                jsonl_keys = [c for c in META_COLUMNS if c != a.text_key and c in meta[0]]
            keep, keep_extra = [], {"ai_score": [], "ai_density": [], "ai_topic": [], "ai_hits": []}
            gen = []
            for i, text in enumerate(texts):
                text = text or ""
                st["docs"] += 1
                n = len(text)
                st["chars_in"] += n
                if n < a.min_chars:
                    continue
                s, strong, dens, counts = score_doc(text)
                if s >= a.min_score and strong >= a.min_strong and dens >= a.min_density:
                    topic, hits = describe(counts)
                    keep.append(i)
                    keep_extra["ai_score"].append(round(s, 2))
                    keep_extra["ai_density"].append(round(dens, 2))
                    keep_extra["ai_topic"].append(topic)
                    keep_extra["ai_hits"].append(hits)
                    topics[topic] += 1
                    st["ai_docs"] += 1
                    st["ai_chars"] += n
                elif a.general_rate > 0 and s < a.min_score / 2 and rng.random() < a.general_rate:
                    gen.append(i)
                    st["general_docs"] += 1
                    st["general_chars"] += n
            if keep:
                out_ai.write(_to_table(texts, meta, keep, {
                    "ai_score": (keep_extra["ai_score"], pa.float32()),
                    "ai_density": (keep_extra["ai_density"], pa.float32()),
                    "ai_topic": (keep_extra["ai_topic"], pa.string()),
                    "ai_hits": (keep_extra["ai_hits"], pa.string())}, a.text_key, jsonl_keys or []))
            if gen:
                out_gen.write(_to_table(texts, meta, gen, {}, a.text_key, jsonl_keys or []))
        out_ai.close()
        out_gen.close()
    except Exception as e:  # noqa: BLE001
        return {"file": job["path"], "error": f"{type(e).__name__}: {e}"}
    res = {"file": job["path"], "root": job["root"], **st, "topics": dict(topics),
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(job["stats"]), exist_ok=True)
    with open(job["stats"] + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(job["stats"] + ".tmp", job["stats"])
    return res


# --------------------------------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------------------------------- #
INPUT_EXT = (".parquet", ".jsonl", ".json", ".jsonl.gz", ".json.gz", ".jsonl.zst", ".jsonl.zstd", ".zst", ".zstd")


def _strip_ext(p: str) -> str:
    for e in sorted(INPUT_EXT, key=len, reverse=True):
        if p.endswith(e):
            return p[:-len(e)]
    return p


def list_jobs(args) -> list[dict]:
    jobs = []
    for pattern in args.input:
        cut = min([pattern.find(ch) for ch in "*?[" if ch in pattern] or [len(pattern)])
        base = os.path.dirname(pattern[:cut]) if cut < len(pattern) else os.path.dirname(pattern)
        root = os.path.basename(base.rstrip("/")) or "input"
        files = sorted(f for f in glob.glob(pattern, recursive=True) if f.endswith(INPUT_EXT) and os.path.isfile(f))
        for f in files:
            rel = _strip_ext(os.path.relpath(f, base))
            jobs.append({"path": f, "root": root, "rel": f"{root}/{rel}",
                         "out_ai": os.path.join(args.out, root, rel + ".parquet"),
                         "out_gen": os.path.join(args.out, "_general", root, rel + ".parquet"),
                         "stats": os.path.join(args.out, "_stats", root, rel + ".json")})
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    return jobs


def dry_run(args, jobs):
    build_automata()
    scores, rows = [], []
    t0, chars = time.time(), 0
    for job in jobs:
        for texts, meta in iter_batches(job["path"], args.text_key):
            for i, text in enumerate(texts):
                text = text or ""
                chars += len(text)
                s, strong, dens, counts = score_doc(text) if len(text) >= args.min_chars else (0.0, 0, 0.0, {})
                url = (meta[i].get("url") if isinstance(meta, list)
                       else (meta.column("url")[i].as_py() if "url" in meta.schema.names else ""))
                scores.append(s)
                rows.append((s, strong, dens, counts, url, text))
                if len(rows) >= args.dry_run:
                    break
            if len(rows) >= args.dry_run:
                break
        if len(rows) >= args.dry_run:
            break
    dt = time.time() - t0
    ok = lambda r: r[0] >= args.min_score and r[1] >= args.min_strong and r[2] >= args.min_density  # noqa: E731
    acc = [r for r in rows if ok(r)]
    near = [r for r in rows if r[0] >= args.min_score / 2 and not ok(r)]
    hit = sorted(s for s in scores if s > 0)
    pct = lambda q: hit[min(len(hit) - 1, int(q * len(hit)))] if hit else 0  # noqa: E731
    print(f"\n{len(rows):,} docs, {chars / 1e6:.1f} M chars in {dt:.1f}s "
          f"({len(rows) / dt:,.0f} docs/s, {chars / dt / 1e6:.1f} MB/s single core, incl. reading)")
    print(f"docs with any hit: {len(hit) / max(1, len(rows)):.1%}   score percentiles of those: "
          f"p50={pct(.5):.1f} p90={pct(.9):.1f} p99={pct(.99):.1f}")
    for t in (4, 6, 8, 10, 15, 20, 30):
        n = sum(1 for r in rows if r[0] >= t and r[1] >= args.min_strong and r[2] >= args.min_density)
        print(f"  min-score {t:>3}: accept {n / max(1, len(rows)):.2%}")
    print(f"ACCEPTED with current thresholds: {len(acc):,} ({len(acc) / max(1, len(rows)):.2%} of docs, "
          f"{sum(len(r[5]) for r in acc) / max(1, chars):.2%} of chars)")

    def show(title, rs):
        print(f"\n--- {title} ---")
        for s, strong, dens, counts, url, text in random.Random(0).sample(rs, min(args.show, len(rs))):
            topic, hits = describe(counts, 6)
            print(f"[score {s:.1f} strong {strong} density {dens:.1f} {topic}] {url}\n   hits: {hits}\n   "
                  f"{' '.join(text[:220].split())}")
    show("accepted (sample)", acc)
    show("near misses: score >= min/2 but rejected (sample)", near)


def summarize(out_dir: str):
    tot, by_root, topics = Counter(), {}, Counter()
    for p in glob.glob(os.path.join(out_dir, "_stats", "**", "*.json"), recursive=True):
        with open(p) as f:
            r = json.load(f)
        root = r.get("root", "?")
        by_root.setdefault(root, Counter()).update({k: v for k, v in r.items() if isinstance(v, int)})
        by_root[root]["files"] += 1
        topics.update(r.get("topics", {}))
    for c in by_root.values():
        tot.update(c)
    summ = {"files": tot["files"], "docs": tot["docs"], "ai_docs": tot["ai_docs"],
            "ai_share_docs": round(tot["ai_docs"] / max(1, tot["docs"]), 5),
            "ai_chars": tot["ai_chars"], "ai_tokens_est": int(tot["ai_chars"] / 4.2),
            "general_docs": tot["general_docs"], "general_tokens_est": int(tot["general_chars"] / 4.2),
            "by_root": {k: {"files": v["files"], "docs": v["docs"], "ai_docs": v["ai_docs"],
                            "ai_tokens_est": int(v["ai_chars"] / 4.2),
                            "general_tokens_est": int(v["general_chars"] / 4.2)} for k, v in by_root.items()},
            "topics": dict(topics.most_common())}
    with open(os.path.join(out_dir, "_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", required=True, help="glob (quote it; ** = recursive); repeatable")
    ap.add_argument("--out", default="ai_cpt")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--shard", default=None, help="i/n: process every n-th file starting at i (multi-server)")
    ap.add_argument("--min-score", type=float, default=12.0)
    ap.add_argument("--min-strong", type=int, default=3, help="distinct terms with weight >= 2")
    ap.add_argument("--min-density", type=float, default=4.0, help="score per 1000 words")
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--general-rate", type=float, default=0.002,
                    help="sample this share of non-AI docs into _general/ for replay (0 = off)")
    ap.add_argument("--dry-run", type=int, default=0, help="score N docs, print stats + examples, write nothing")
    ap.add_argument("--show", type=int, default=8, help="examples per group in --dry-run")
    args = ap.parse_args()

    jobs = list_jobs(args)
    if not jobs:
        sys.exit(f"no input files match {args.input}")
    if args.dry_run:
        return dry_run(args, jobs)

    todo = [j for j in jobs if not os.path.exists(j["stats"])]
    print(f"{len(jobs):,} files ({len(jobs) - len(todo):,} already done), {args.workers} workers -> {args.out}",
          flush=True)
    for j in todo:
        j["args"] = args
    t0, done, chars, ai_docs, docs, errors = time.time(), 0, 0, 0, 0, []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futs = [ex.submit(process_file, j) for j in todo]
        for fu in as_completed(futs):
            r = fu.result()
            done += 1
            if "error" in r:
                errors.append(r)
                print(f"ERROR {r['file']}: {r['error']}", flush=True)
                continue
            chars += r.get("chars_in", 0)
            docs += r.get("docs", 0)
            ai_docs += r.get("ai_docs", 0)
            dt = time.time() - t0
            print(f"[{done:,}/{len(todo):,}] {r['file']}  ai {r.get('ai_docs', 0):,}/{r.get('docs', 0):,}  "
                  f"| total ai {ai_docs:,}/{docs:,} ({ai_docs / max(1, docs):.2%})  "
                  f"{chars / dt / 1e6:,.0f} MB/s  eta {(len(todo) - done) * dt / done / 3600:.1f} h", flush=True)
    summ = summarize(args.out)
    print(json.dumps({k: v for k, v in summ.items() if k != "topics"}, indent=2))
    print("topics:", summ["topics"])
    if errors:
        print(f"{len(errors)} files failed (rerun the same command to retry them)")


if __name__ == "__main__":
    main()
