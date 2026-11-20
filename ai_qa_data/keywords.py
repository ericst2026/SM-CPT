"""Weighted keyword / regex scorer for AI relevance.

Three independent scores are produced:
  * core  - AI / ML / DL / data-science vocabulary (concepts, history, ethics, apps)
  * math  - math that matters for AI derivations (lin-alg, prob/stats, calculus, optimisation, info theory)
  * code  - Python ML ecosystem (torch, sklearn, numpy, ...)

Each term has a weight:
  3 = unambiguous AI term ("backpropagation", "large language model")
  2 = strong but occasionally used elsewhere ("classifier", "dataset")
  1 = weak / ambiguous, only meaningful together with other hits ("model", "agent")

The scorer is pure-regex and runs at ~10-50k rows/s per core, so it is used both as
a pre-filter while streaming huge datasets and as stage 2 of the splitter.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Lexicons:  (regex, weight, topic_hint or None)
# Patterns are matched case-insensitively unless listed in *_CS (case-sensitive).
# --------------------------------------------------------------------------- #
CORE_TERMS: list[tuple[str, float, str | None]] = [
    # --- field names ---
    (r"artificial[\s-]+intelligence", 3, None),
    (r"machine[\s-]+learning", 3, None),
    (r"deep[\s-]+learning", 3, None),
    (r"reinforcement[\s-]+learning", 3, None),
    (r"(?:un|semi[\s-]?|self[\s-]?)?supervised[\s-]+learning", 3, "concepts"),
    (r"natural[\s-]+language[\s-]+processing", 3, None),
    (r"computer[\s-]+vision", 3, None),
    (r"data[\s-]+scien(?:ce|tists?)", 2, None),
    (r"data[\s-]+mining", 2, None),
    (r"pattern[\s-]+recognition", 2, None),
    (r"generative[\s-]+(?:ai|model|models|pre-?trained)", 3, None),
    (r"speech[\s-]+recognition", 2, "applications"),
    (r"robotics?", 1, "applications"),
    # --- models / architectures ---
    (r"neural[\s-]+net(?:work)?s?", 3, None),
    (r"(?:artificial|biological)[\s-]+neurons?", 2, "concepts"),
    (r"perceptrons?", 3, None),
    (r"multi[\s-]?layer[\s-]+perceptron", 3, None),
    (r"convolutional", 3, None),
    (r"recurrent[\s-]+neural", 3, None),
    (r"long[\s-]+short[\s-]+term[\s-]+memory", 3, None),
    (r"transformer[\s-]+(?:model|architecture|block|layer|encoder|decoder)s?", 3, None),
    (r"(?:self|cross|multi[\s-]?head)[\s-]+attention", 3, None),
    (r"attention[\s-]+(?:mechanism|head|weights?|layer)s?", 3, None),
    (r"large[\s-]+language[\s-]+models?", 3, None),
    (r"language[\s-]+models?", 2, None),
    (r"foundation[\s-]+models?", 3, None),
    (r"diffusion[\s-]+models?", 3, None),
    (r"generative[\s-]+adversarial", 3, None),
    (r"(?:variational[\s-]+)?auto[\s-]?encoders?", 3, None),
    (r"word[\s-]?(?:2vec|embeddings?)|glove[\s-]+embeddings?", 3, None),
    (r"embeddings?(?:[\s-]+(?:vector|space|layer|model)s?)?", 1, None),
    (r"random[\s-]+forests?", 3, None),
    (r"decision[\s-]+trees?", 2, None),
    (r"gradient[\s-]+boost(?:ing|ed)", 3, None),
    (r"support[\s-]+vector[\s-]+machines?", 3, None),
    (r"k[\s-]?nearest[\s-]+neighbou?rs?", 3, None),
    (r"k[\s-]?means(?:[\s-]+clustering)?", 3, None),
    (r"naive[\s-]+bayes", 3, None),
    (r"logistic[\s-]+regression", 3, None),
    (r"linear[\s-]+regression", 2, None),
    (r"hidden[\s-]+markov[\s-]+models?", 3, None),
    (r"markov[\s-]+decision[\s-]+process(?:es)?", 3, None),
    (r"q[\s-]?learning|policy[\s-]+gradients?|actor[\s-]?critic", 3, None),
    (r"mixture[\s-]+of[\s-]+experts", 3, None),
    (r"retrieval[\s-]+augmented[\s-]+generation", 3, None),
    (r"knowledge[\s-]+graphs?", 2, None),
    (r"expert[\s-]+systems?", 2, "history"),
    (r"chat[\s-]?bots?", 2, "applications"),
    (r"(?:ai|intelligent|autonomous|software)[\s-]+agents?", 2, None),
    # --- ML frameworks & tools (as words in prose, not code) ---
    (r"pytorch|tensorflow|keras|scikit[\s-]?learn|sklearn|hugging[\s-]?face|xgboost|lightgbm|jax|mlflow|weights[\s-]+(?:&|and)[\s-]+biases|wandb|onnx|tensorrt|vllm|langchain|llama[\s-]?index|opencv|nltk|spacy", 3, "coding"),
    (r"jupyter(?:[\s-]+notebooks?)?|kaggle|colab", 1, "coding"),
    # --- training / evaluation vocabulary ---
    (r"back[\s-]?propagation", 3, "theory_math"),
    (r"(?:stochastic[\s-]+)?gradient[\s-]+descent", 3, "theory_math"),
    (r"loss[\s-]+functions?", 3, None),
    (r"cost[\s-]+functions?", 1, None),
    (r"cross[\s-]?entropy", 3, "theory_math"),
    (r"activation[\s-]+functions?", 3, None),
    (r"relu|leaky[\s-]+relu|gelu", 3, None),
    (r"soft[\s-]?max", 3, "theory_math"),
    (r"drop[\s-]?out(?:[\s-]+(?:layer|rate|regularization))?", 1, None),
    (r"batch[\s-]+norm(?:alization)?|layer[\s-]+norm(?:alization)?", 3, None),
    (r"over[\s-]?fitting|under[\s-]?fitting", 3, "concepts"),
    (r"bias[\s-]+variance", 3, "concepts"),
    (r"hyper[\s-]?parameters?", 3, None),
    (r"learning[\s-]+rate", 3, None),
    (r"(?:training|validation|test)[\s-]+(?:set|data|loss|accuracy|split)s?", 2, None),
    (r"train(?:ing)?[\s-]+(?:a|the|your)?[\s-]*models?", 2, None),
    (r"fine[\s-]?tun(?:e|ed|ing)", 3, None),
    (r"pre[\s-]?train(?:ed|ing)", 2, None),
    (r"transfer[\s-]+learning", 3, None),
    (r"feature[\s-]+(?:engineering|extraction|selection|vectors?|maps?)", 2, None),
    (r"dimensionality[\s-]+reduction|principal[\s-]+component[\s-]+analysis|t-?sne|umap", 3, None),
    (r"regulari[sz]ation", 2, None),
    (r"cross[\s-]?validation", 2, None),
    (r"confusion[\s-]+matrix|precision[\s-]+and[\s-]+recall|f1[\s-]?score|roc[\s-]+curves?|auc[\s-]+score", 3, None),
    (r"classifiers?|classification[\s-]+(?:model|task|problem|accuracy)s?", 2, None),
    (r"clustering(?:[\s-]+algorithms?)?", 2, None),
    (r"labell?ed[\s-]+data|unlabell?ed[\s-]+data|data[\s-]+labell?ing", 2, None),
    (r"datasets?", 1, None),
    (r"epochs?", 1, None),
    (r"inference[\s-]+(?:time|speed|server|engine)", 2, None),
    (r"tokeni[sz](?:er|ers|ation|ed)", 2, None),
    (r"prompt[\s-]+engineering|in[\s-]?context[\s-]+learning|chain[\s-]?of[\s-]?thought", 3, None),
    (r"hallucinat(?:e|es|ion|ions)", 1, "ethics_society"),
    # --- applications ---
    (r"self[\s-]?driving|autonomous[\s-]+(?:vehicles?|cars?|driving)", 2, "applications"),
    (r"recommend(?:ation|er)[\s-]+(?:systems?|engines?|algorithms?)", 3, "applications"),
    (r"facial[\s-]+recognition|face[\s-]+recognition|image[\s-]+recognition|object[\s-]+detection", 3, "applications"),
    (r"image[\s-]+(?:classification|segmentation|generation)|semantic[\s-]+segmentation", 3, "applications"),
    (r"machine[\s-]+translation|sentiment[\s-]+analysis|named[\s-]+entity[\s-]+recognition|text[\s-]+classification", 3, "applications"),
    (r"virtual[\s-]+assistants?|voice[\s-]+assistants?|siri|alexa", 1, "applications"),
    (r"fraud[\s-]+detection|anomaly[\s-]+detection|predictive[\s-]+(?:model(?:ing|s)?|analytics|maintenance)", 2, "applications"),
    (r"medical[\s-]+imaging|drug[\s-]+discovery|protein[\s-]+folding|alphafold", 1, "applications"),
    # --- history ---
    (r"turing[\s-]+test|alan[\s-]+turing", 3, "history"),
    (r"dartmouth[\s-]+(?:workshop|conference|summer)", 3, "history"),
    (r"ai[\s-]+winters?", 3, "history"),
    (r"(?:john[\s-]+mccarthy|marvin[\s-]+minsky|frank[\s-]+rosenblatt|geoffrey[\s-]+hinton|yann[\s-]+lecun|yoshua[\s-]+bengio|andrew[\s-]+ng|fei[\s-]?fei[\s-]+li|demis[\s-]+hassabis)", 3, "history"),
    (r"eliza|deep[\s-]+blue|alphago|alphazero|watson[\s-]+(?:computer|jeopardy)|imagenet|alexnet", 2, "history"),
    (r"lighthill[\s-]+report|symbolic[\s-]+ai|connectionis[mt]|good[\s-]+old[\s-]+fashioned[\s-]+ai", 3, "history"),
    # --- ethics / society ---
    (r"(?:algorithmic|ai|model|data)[\s-]+bias|fairness[\s-]+in[\s-]+(?:ai|machine)", 3, "ethics_society"),
    (r"ai[\s-]+(?:safety|alignment|ethics|regulation|governance|act|policy|risk)", 3, "ethics_society"),
    (r"deep[\s-]?fakes?", 3, "ethics_society"),
    (r"explainab(?:le|ility)|interpretab(?:le|ility)", 2, "ethics_society"),
    (r"(?:job|work(?:er)?)[\s-]+(?:displacement|automation)|automation[\s-]+of[\s-]+jobs", 1, "ethics_society"),
    # --- research ---
    (r"state[\s-]of[\s-]the[\s-]art|sota|benchmark(?:s|ed|ing)?", 1, "research"),
    (r"arxiv|neurips|icml|iclr|cvpr|acl[\s-]+anthology|emnlp|aaai", 2, "research"),
    (r"ablation|we[\s-]+propose", 1, "research"),

    # ===================== v0.2 expansion: every major AI / ML area =====================
    # --- learning paradigms ---
    (r"(?:few|zero|one)[\s-]?shot[\s-]+(?:learning|learners?|classification|prompting)", 3, "concepts"),
    (r"meta[\s-]?learning|learning[\s-]+to[\s-]+learn|multi[\s-]?task[\s-]+learning", 3, "concepts"),
    (r"(?:active|online|incremental|continual|lifelong|curriculum|contrastive|representation|metric|manifold|ensemble|"
     r"federated|imitation|inverse[\s-]+reinforcement|multi[\s-]?instance|weakly[\s-]+supervised)[\s-]+learning", 3, "concepts"),
    (r"catastrophic[\s-]+forgetting|domain[\s-]+(?:adaptation|shift|generali[sz]ation)|distribution[\s-]+shift|"
     r"(?:data|concept|covariate)[\s-]+drift|out[\s-]of[\s-]distribution", 3, "concepts"),
    (r"self[\s-]?supervised|semi[\s-]?supervised|unsupervised|supervised[\s-]+(?:model|method|algorithm|task)s?", 2, "concepts"),
    (r"inductive[\s-]+bias|no[\s-]+free[\s-]+lunch|curse[\s-]+of[\s-]+dimensionality|vc[\s-]+dimension|pac[\s-]+learning|"
     r"universal[\s-]+approximation", 3, "theory_math"),
    # --- classical ML algorithms ---
    (r"ensemble[\s-]+(?:methods?|models?)|bagging[\s-]+(?:and[\s-]+boosting|classifiers?|ensembles?)|bootstrap[\s-]+aggregat\w+|adaboost|boosting[\s-]+algorithms?|"
     r"stacking[\s-]+(?:ensemble|classifier|regressor)", 3, None),
    (r"kernel[\s-]+(?:trick|methods?|svm|ridge|pca|density[\s-]+estimation)|radial[\s-]+basis[\s-]+functions?", 3, "theory_math"),
    (r"decision[\s-]+(?:boundar(?:y|ies)|stumps?)|gini[\s-]+(?:impurity|index)|information[\s-]+gain", 3, None),
    (r"(?:gaussian|dirichlet)[\s-]+(?:mixture[\s-]+models?|process(?:es)?)|expectation[\s-]+maximi[sz]ation|em[\s-]+algorithm", 3, "theory_math"),
    (r"dbscan|hierarchical[\s-]+clustering|spectral[\s-]+clustering|agglomerative[\s-]+clustering|mean[\s-]+shift[\s-]+clustering", 3, None),
    (r"(?:ridge|lasso|elastic[\s-]+net|polynomial|softmax|multinomial[\s-]+logistic)[\s-]+regression", 3, None),
    (r"perceptron[\s-]+learning|hebbian[\s-]+learning|self[\s-]?organi[sz]ing[\s-]+maps?|boltzmann[\s-]+machines?|hopfield[\s-]+networks?", 3, "history"),
    (r"(?:bayesian|belief)[\s-]+networks?|probabilistic[\s-]+graphical[\s-]+models?|markov[\s-]+random[\s-]+fields?|"
     r"conditional[\s-]+random[\s-]+fields?|variational[\s-]+inference|latent[\s-]+(?:variables?|space|dirichlet[\s-]+allocation)", 3, "theory_math"),
    (r"collaborative[\s-]+filtering|content[\s-]+based[\s-]+filtering|matrix[\s-]+factori[sz]ation[\s-]+(?:for|in)[\s-]+recommend\w*", 3, "applications"),
    (r"time[\s-]+series[\s-]+(?:forecasting|classification|anomaly)|forecasting[\s-]+models?", 2, "applications"),
    (r"feature[\s-]+(?:importance|scaling|normali[sz]ation|crosses|store)|one[\s-]?hot[\s-]+encod\w+|label[\s-]+encod\w+|"
     r"imputation|class[\s-]+imbalance|imbalanced[\s-]+(?:data|dataset|classes)|smote|data[\s-]+augmentation", 2, None),
    (r"(?:mean[\s-]+absolute|root[\s-]+mean[\s-]+squared)[\s-]+error|log[\s-]+loss|hinge[\s-]+loss|"
     r"(?:top[\s-]?[15]|balanced)[\s-]+accuracy|perplexity[\s-]+(?:score|of[\s-]+the[\s-]+(?:language[\s-]+)?model)|bleu[\s-]+score|rouge[\s-]?(?:score|[12l])|mean[\s-]+average[\s-]+precision|"
     r"intersection[\s-]+over[\s-]+union", 3, None),
    (r"grid[\s-]+search|random[\s-]+search|bayesian[\s-]+optimi[sz]ation|hyperparameter[\s-]+(?:tuning|search|optimi[sz]ation)|"
     r"early[\s-]+stopping|k[\s-]?fold", 3, None),
    # --- deep learning: architectures, training, optimisation ---
    (r"(?:fully[\s-]+connected|dense|hidden|pooling|embedding|output|input)[\s-]+layers?|deep[\s-]+neural|"
     r"feed[\s-]?forward[\s-]+(?:neural[\s-]+)?networks?|skip[\s-]+connections?|residual[\s-]+(?:connections?|blocks?|networks?)", 3, None),
    (r"(?:vanishing|exploding)[\s-]+gradients?|gradient[\s-]+(?:clipping|checkpointing|accumulation)|weight[\s-]+(?:decay|initiali[sz]ation|sharing)|"
     r"xavier[\s-]+initiali[sz]ation|he[\s-]+initiali[sz]ation|mixed[\s-]+precision[\s-]+training|mini[\s-]?batch(?:es)?", 3, None),
    (r"adam(?:w)?[\s-]+optimi[sz]er|rmsprop|adagrad|momentum[\s-]+(?:optimi[sz]er|sgd)|learning[\s-]+rate[\s-]+(?:schedul\w+|warm[\s-]?up|decay)|"
     r"cosine[\s-]+annealing", 3, "theory_math"),
    (r"encoder[\s-]?decoder|seq(?:uence)?[\s-]?(?:2|to)[\s-]?seq(?:uence)?|sequence[\s-]+models?|positional[\s-]+(?:encoding|embedding)s?|"
     r"(?:rotary|relative)[\s-]+position(?:al)?[\s-]+(?:encoding|embedding)s?|scaled[\s-]+dot[\s-]?product|query,?[\s-]+key,?[\s-]+(?:and[\s-]+)?value", 3, None),
    (r"(?:graph|spiking|capsule|siamese|bayesian|liquid|physics[\s-]?informed|kolmogorov[\s-]?arnold)[\s-]+neural[\s-]+networks?|"
     r"graph[\s-]+(?:convolution\w*|attention[\s-]+networks?|embeddings?)|message[\s-]+passing[\s-]+neural|node[\s-]+(?:classification|embeddings?)", 3, None),
    (r"normali[sz]ing[\s-]+flows?|energy[\s-]+based[\s-]+models?|score[\s-]+(?:matching|based[\s-]+models?)|denoising[\s-]+(?:diffusion|auto[\s-]?encoders?)|"
     r"latent[\s-]+diffusion|flow[\s-]+matching|autoregressive[\s-]+models?|next[\s-]+token[\s-]+prediction", 3, None),
    (r"state[\s-]+space[\s-]+models?|mamba[\s-]+(?:model|architecture)|vision[\s-]+transformers?|neural[\s-]+architecture[\s-]+search|"
     r"neural[\s-]+radiance[\s-]+fields?|gaussian[\s-]+splatting", 3, None),
    (r"model[\s-]+(?:compression|distillation|quantization|pruning|parallelism|merging|collapse|weights|parameters|checkpoints?|"
     r"architecture|capacity|interpretability)|knowledge[\s-]+distillation|(?:post[\s-]?training|weight)[\s-]+quantization|"
     r"(?:int[48]|4[\s-]?bit|8[\s-]?bit)[\s-]+quantization", 3, None),
    (r"(?:data|tensor|pipeline|expert)[\s-]+parallelism|kv[\s-]+cache|flash[\s-]?attention|speculative[\s-]+decoding|"
     r"(?:greedy|beam)[\s-]+(?:search|decoding)|top[\s-]?[kp][\s-]+sampling|nucleus[\s-]+sampling|temperature[\s-]+sampling", 3, None),
    # --- LLMs / generative AI ---
    (r"(?:instruction|supervised)[\s-]+(?:fine[\s-]?)?tuning|parameter[\s-]+efficient[\s-]+fine[\s-]?tuning|low[\s-]?rank[\s-]+adaptation|"
     r"reinforcement[\s-]+learning[\s-]+from[\s-]+human[\s-]+feedback|preference[\s-]+(?:optimi[sz]ation|data|tuning)|reward[\s-]+models?|"
     r"direct[\s-]+preference|constitutional[\s-]+ai", 3, None),
    (r"context[\s-]+(?:window|length)|system[\s-]+prompts?|prompt[\s-]+(?:injection|templates?|tuning)|few[\s-]?shot[\s-]+examples?|"
     r"jailbreak(?:s|ing)?[\s-]+(?:an?[\s-]+)?(?:llm|model|chatbot|ai)|(?:ai|llm|model)[\s-]+jailbreak(?:s|ing)?|guardrails?[\s-]+for", 3, None),
    (r"vector[\s-]+(?:databases?|stores?|search|embeddings?)|semantic[\s-]+search|sentence[\s-]+embeddings?|text[\s-]+embeddings?|"
     r"(?:dense|sparse)[\s-]+retrieval|re[\s-]?rank(?:er|ing)|chunking[\s-]+strateg\w+", 3, None),
    (r"(?:ai|llm)[\s-]+agents?|agentic|tool[\s-]+calling|tool[\s-]+use[\s-]+(?:in|by|for)[\s-]+(?:llms?|agents?|models?)|function[\s-]+calling|multi[\s-]?agent[\s-]+systems?|"
     r"react[\s-]+(?:prompting|agent|framework)|tree[\s-]+of[\s-]+thoughts?|self[\s-]?consistency|reasoning[\s-]+models?", 3, None),
    (r"text[\s-]+to[\s-]+(?:image|video|speech|3d)|image[\s-]+to[\s-]+text|image[\s-]+captioning|visual[\s-]+question[\s-]+answering|"
     r"(?:vision|visual)[\s-]+language[\s-]+models?|multimodal[\s-]+(?:models?|llms?|learning|ai)|speech[\s-]+synthesis|"
     r"voice[\s-]+cloning|style[\s-]+transfer|super[\s-]?resolution[\s-]+(?:model|network)|inpainting", 3, "applications"),
    (r"neural[\s-]+scaling[\s-]+laws?|scaling[\s-]+laws?[\s-]+(?:for|of)[\s-]+(?:language|neural|llms?|ai|large|deep)|emergent[\s-]+(?:abilities|capabilities|behaviou?r)|in[\s-]?context[\s-]+examples?|"
     r"synthetic[\s-]+(?:training[\s-]+)?data[\s-]+generation|data[\s-]+contamination|benchmark[\s-]+contamination", 3, "research"),
    # --- NLP ---
    (r"part[\s-]of[\s-]speech[\s-]+tagg\w+|dependency[\s-]+pars\w+|constituency[\s-]+pars\w+|word[\s-]+sense[\s-]+disambiguation|"
     r"coreference[\s-]+resolution|question[\s-]+answering[\s-]+(?:system|model)s?|text[\s-]+(?:summari[sz]ation|generation)|"
     r"information[\s-]+extraction|topic[\s-]+model(?:l?ing|s)?|n[\s-]?gram[\s-]+(?:language[\s-]+)?models?|bag[\s-]+of[\s-]+words|tf[\s-]?idf|"
     r"byte[\s-]+pair[\s-]+encoding|word[\s-]?piece|subword[\s-]+tokeni[sz]\w+|lemmati[sz]ation", 3, None),
    (r"speech[\s-]+(?:to[\s-]+text|recognition[\s-]+models?)|automatic[\s-]+speech[\s-]+recognition|text[\s-]+to[\s-]+speech|"
     r"speaker[\s-]+(?:recognition|diari[sz]ation)|keyword[\s-]+spotting", 3, "applications"),
    # --- computer vision ---
    (r"(?:instance|panoptic)[\s-]+segmentation|pose[\s-]+estimation|optical[\s-]+character[\s-]+recognition|optical[\s-]+flow|"
     r"(?:bounding|anchor)[\s-]+box(?:es)?|non[\s-]?maximum[\s-]+suppression|feature[\s-]+pyramid|image[\s-]+(?:embedding|retrieval|synthesis)|"
     r"depth[\s-]+estimation|visual[\s-]+(?:odometry|slam|recognition)|convolution(?:al)?[\s-]+(?:filters?|kernels?|layers?)", 3, "applications"),
    # --- reinforcement learning ---
    (r"bellman[\s-]+(?:equation|operator|backup)|value[\s-]+iteration|(?:state|action)[\s-]+value[\s-]+functions?|policy[\s-]+(?:iteration|optimi[sz]ation|network)|"
     r"temporal[\s-]+difference|td[\s-]?(?:learning|error|\(\s*lambda\s*\))|(?:deep[\s-]+)?q[\s-]?networks?|reward[\s-]+(?:function|shaping|signal|hacking)|"
     r"exploration[\s-]+(?:vs\.?|versus|and)[\s-]+exploitation|epsilon[\s-]?greedy|multi[\s-]?armed[\s-]+bandits?|"
     r"(?:model[\s-]?based|model[\s-]?free|off[\s-]?policy|on[\s-]?policy|offline)[\s-]+(?:reinforcement[\s-]+learning|rl)|"
     r"proximal[\s-]+policy[\s-]+optimi[sz]ation|monte[\s-]+carlo[\s-]+tree[\s-]+search|self[\s-]?play|"
     r"state[\s-]+and[\s-]+action[\s-]+spaces?|action[\s-]+space", 3, None),
    # --- classical / symbolic AI ---
    (r"a\*[\s-]+(?:search|algorithm)|best[\s-]+first[\s-]+search|heuristic[\s-]+(?:search|function)|admissible[\s-]+heuristic|"
     r"minimax|alpha[\s-]?beta[\s-]+pruning|game[\s-]+tree[\s-]+search|adversarial[\s-]+search|constraint[\s-]+satisfaction|"
     r"(?:automated|classical|hierarchical)[\s-]+planning|strips[\s-]+planning|pddl|knowledge[\s-]+(?:representation|engineering)|"
     r"(?:forward|backward)[\s-]+chaining|inference[\s-]+engine|production[\s-]+rules|semantic[\s-]+(?:networks?|web)|ontolog(?:y|ies)[\s-]+(?:in|for)[\s-]+ai|"
     r"fuzzy[\s-]+(?:logic|sets?|inference)|(?:genetic|evolutionary)[\s-]+(?:algorithms?|programming|computation)|"
     r"swarm[\s-]+intelligence|particle[\s-]+swarm|ant[\s-]+colony|simulated[\s-]+annealing|neuro[\s-]?symbolic|"
     r"intelligent[\s-]+(?:systems?|tutoring)|(?:simple|model)[\s-]+reflex[\s-]+agents?|peas[\s-]+description", 3, "concepts"),
    (r"(?:general|narrow|weak|strong|superintelligent|human[\s-]?level)[\s-]+(?:ai|artificial[\s-]+intelligence)|superintelligence|"
     r"artificial[\s-]+general[\s-]+intelligence|chinese[\s-]+room|symbol[\s-]+grounding|frame[\s-]+problem|"
     r"technological[\s-]+singularity|machine[\s-]+(?:intelligence|consciousness|reasoning|perception|ethics)", 3, "concepts"),
    # --- robotics / autonomous systems ---
    (r"(?:robot|motion|path|trajectory)[\s-]+(?:planning|learning|control|perception)|simultaneous[\s-]+locali[sz]ation[\s-]+and[\s-]+mapping|"
     r"sensor[\s-]+fusion|sim[\s-]?to[\s-]?real|embodied[\s-]+(?:ai|agents?)|humanoid[\s-]+robots?|robotic[\s-]+(?:arm|manipulation|process[\s-]+automation)", 3, "applications"),
    # --- MLOps / engineering ---
    (r"mlops|llmops|(?:model|ml)[\s-]+(?:deployment|serving|monitoring|registry|pipeline|lifecycle|ops|inference)|"
     r"feature[\s-]+stores?|experiment[\s-]+tracking|inference[\s-]+(?:latency|optimi[sz]ation|endpoint)|"
     r"(?:training|ml)[\s-]+pipelines?|automl|auto[\s-]?ml|edge[\s-]+ai|on[\s-]?device[\s-]+(?:ml|ai|inference)|tinyml|"
     r"(?:ai|ml)[\s-]+(?:accelerators?|chips?|hardware|infrastructure|compute)|neural[\s-]+processing[\s-]+units?", 3, "coding"),
    # --- data science ---
    (r"exploratory[\s-]+data[\s-]+analysis|data[\s-]+(?:preprocessing|cleaning|wrangling|pipeline|annotation|labell?ers?)|"
     r"training[\s-]+(?:examples?|samples?|corpus|run|loop|objective)|ground[\s-]+truth[\s-]+labels?|predictive[\s-]+analytics|"
     r"big[\s-]+data[\s-]+analytics|(?:a/b|ab)[\s-]+test(?:ing)?[\s-]+(?:of|for)[\s-]+models?", 2, None),
    # --- trustworthy / responsible AI ---
    (r"adversarial[\s-]+(?:examples?|attacks?|robustness|training|perturbations?)|data[\s-]+poisoning|backdoor[\s-]+attacks?|"
     r"model[\s-]+(?:inversion|extraction|stealing)|membership[\s-]+inference|differential[\s-]+privacy|"
     r"shap(?:ley)?[\s-]+values?|lime[\s-]+explanations?|saliency[\s-]+maps?|grad[\s-]?cam|feature[\s-]+attribution|"
     r"mechanistic[\s-]+interpretability|sparse[\s-]+auto[\s-]?encoders?|(?:ai|outer|inner)[\s-]+alignment|"
     r"specification[\s-]+gaming|existential[\s-]+risk[\s-]+from[\s-]+ai|responsible[\s-]+ai|trustworthy[\s-]+ai|ai[\s-]+(?:audits?|accountability|"
     r"transparency|literacy|art|generated|detection|watermark\w*|copyright)|eu[\s-]+ai[\s-]+act|disparate[\s-]+impact|"
     r"(?:demographic|statistical)[\s-]+parity|equali[sz]ed[\s-]+odds|algorithmic[\s-]+(?:fairness|accountability|decision[\s-]?making)", 3, "ethics_society"),
    # --- applications ---
    (r"(?:ai|machine[\s-]+learning|ml|deep[\s-]+learning)[\s-]+(?:in|for)[\s-]+(?:healthcare|medicine|finance|education|agriculture|"
     r"climate|law|marketing|manufacturing|cybersecurity|security|science|drug)|(?:ai|ml)[\s-]?(?:powered|driven|based|enabled)|"
     r"credit[\s-]+scoring[\s-]+models?|algorithmic[\s-]+trading|spam[\s-]+(?:filter|detection|classifier)|"
     r"churn[\s-]+prediction|demand[\s-]+forecasting|intrusion[\s-]+detection", 2, "applications"),
    # --- history (additional) ---
    (r"(?:geoffrey|geoff)[\s-]+hinton|j[uü]rgen[\s-]+schmidhuber|ilya[\s-]+sutskever|claude[\s-]+shannon|herbert[\s-]+simon|allen[\s-]+newell|"
     r"judea[\s-]+pearl|richard[\s-]+sutton|ian[\s-]+goodfellow|andrej[\s-]+karpathy|stuart[\s-]+russell|peter[\s-]+norvig|"
     r"logic[\s-]+theorist|general[\s-]+problem[\s-]+solver|shakey[\s-]+the[\s-]+robot|mycin|dendral|lisp[\s-]+machines?|"
     r"fifth[\s-]+generation[\s-]+computer|attention[\s-]+is[\s-]+all[\s-]+you[\s-]+need|perceptrons[\s-]+book", 3, "history"),
]

# Case-sensitive acronyms / product names (avoid "ai" in other words, "ml" = millilitre, ...)
CORE_TERMS_CS: list[tuple[str, float, str | None]] = [
    (r"AI", 2, None),
    (r"A\.I\.", 2, None),
    (r"ML", 2, None),
    (r"NLP", 3, None),
    (r"LLMs?", 3, None),
    (r"GPT(?:-?[1-5](?:\.\d)?o?)?|ChatGPT|GPT-4o", 3, None),
    (r"BERT|RoBERTa|DistilBERT|LLaMA|Llama[\s-]?[1-4]|Qwen\d?(?:\.\d)?|DeepSeek", 2, None),
    (r"T5|Mistral|Gemini|Claude|Copilot|Midjourney|Stable\s+Diffusion|DALL-?E", 1, None),
    (r"CNNs?|RNNs?|LSTMs?|GRUs?|GANs?|VAEs?|MLPs?|SVMs?|KNN|PCA|MDP|RLHF|DPO|PPO|GRPO|LoRA|QLoRA|RAG|ViT|YOLO(?:v\d+)?|ResNet|U-Net|AGI|XAI", 3, None),
    (r"GPUs?|TPUs?|CUDA", 1, None),
    # v0.2 expansion
    (r"GNNs?|GCNs?|DQN|DDPG|A3C|TD3|TRPO|MCTS|VLMs?|LVLMs?|MLLMs?|PEFT|KV-?cache|RLAIF|GBDT|AutoML|MLOps|LLMOps|SMOTE|"
     r"t-SNE|UMAP|NeRF|DETR|GPTs|EU AI Act|XGBoost|LightGBM|CatBoost|OpenAI|Anthropic|DeepMind|Hugging Face|Mixtral|"
     r"AlphaFold|AlphaGo|AlphaZero|AlphaStar|MuZero", 3, None),
    # acronyms that also mean other things (SAM, NAS, SAC, ASR, ...): need company before they count
    (r"GATs?|A2C|SAC|SLAM|OCR|ASR|TTS|NER|MoE|SFT|BPE|CoT|ReAct|NAS|CLIP|SigLIP|SAM|BLEU|ROUGE|IoU|mAP|SHAP|SLMs?|ASI|"
     r"Mistral AI|Meta AI|Stability AI|Perplexity AI|Gemma|Phi-[1-4]", 2, None),
    (r"Transformers?|Attention|Embeddings?|Agents?|Copilots?|Cohere|xAI|Grok|Falcon|Sora|Whisper|Codex|Watson", 1, None),
]

MATH_TERMS: list[tuple[str, float, str | None]] = [
    # linear algebra
    (r"matri(?:x|ces)", 2, None),
    (r"eigen(?:value|vector|decomposition)s?", 3, None),
    (r"singular[\s-]+value(?:[\s-]+decomposition)?|\bsvd\b", 3, None),
    (r"dot[\s-]+products?|inner[\s-]+products?|outer[\s-]+products?", 2, None),
    (r"vectors?(?:[\s-]+spaces?)?|linear[\s-]+(?:transformation|map|independence|combination)s?", 1, None),
    (r"determinants?|trace[\s-]+of|rank[\s-]+of[\s-]+(?:a|the)[\s-]+matrix|orthogonal|projection", 1, None),
    (r"norm(?:s)?[\s-]+of|l[12][\s-]+norm|euclidean[\s-]+(?:distance|norm)|cosine[\s-]+similarity", 2, None),
    # calculus / optimisation
    (r"gradients?", 2, None),
    (r"partial[\s-]+derivatives?|chain[\s-]+rule|jacobian|hessian", 3, None),
    (r"derivatives?|differentiat(?:e|ion)", 1, None),
    (r"convex(?:ity)?|concave|lagrange[\s-]+multipliers?|kkt", 2, None),
    (r"minimi[sz](?:e|es|ation|ing)|maximi[sz](?:e|es|ation|ing)|optimi[sz]ation|local[\s-]+minim(?:a|um)|global[\s-]+minim(?:a|um)", 1, None),
    (r"taylor[\s-]+(?:series|expansion)", 1, None),
    (r"sigmoid|logistic[\s-]+function|softmax|tanh|relu", 3, None),
    (r"least[\s-]+squares|mean[\s-]+squared[\s-]+error|\bmse\b", 3, None),
    # probability / statistics
    (r"probabilit(?:y|ies)", 1, None),
    (r"conditional[\s-]+probabilit(?:y|ies)|bayes(?:'|ian)?(?:[\s-]+(?:theorem|rule|inference|network))?|posterior|prior[\s-]+distribution", 3, None),
    (r"expect(?:ed[\s-]+value|ation)s?|varian(?:ce|ces)|covarian(?:ce|ces)|standard[\s-]+deviation", 2, None),
    (r"random[\s-]+variables?|probability[\s-]+(?:distribution|density|mass)|(?:normal|gaussian|bernoulli|binomial|poisson|exponential|uniform|beta|dirichlet|multinomial)[\s-]+distributions?", 2, None),
    (r"(?:maximum|log)[\s-]+likelihood|\bmle\b|\bmap\s+estimat", 3, None),
    (r"entropy|kl[\s-]+divergence|kullback|mutual[\s-]+information", 3, None),
    (r"markov[\s-]+chains?|stationary[\s-]+distribution|monte[\s-]+carlo", 2, None),
    (r"hypothesis[\s-]+test(?:ing)?|p[\s-]?values?|confidence[\s-]+intervals?|correlation", 1, None),
    (r"regression", 2, None),
]

CODE_TERMS: list[tuple[str, float, str | None]] = [
    (r"import\s+torch|from\s+torch|torch\.\w+|nn\.Module|\.backward\(\)|optimizer\.step\(\)|DataLoader", 3, None),
    (r"import\s+tensorflow|from\s+tensorflow|tf\.keras|import\s+keras|from\s+keras|model\.compile\(", 3, None),
    (r"from\s+sklearn|import\s+sklearn|scikit[\s-]?learn|train_test_split|\.fit\(|\.predict\(|\.fit_transform\(", 3, None),
    (r"import\s+jax|from\s+jax|import\s+flax|import\s+optax", 3, None),
    (r"from\s+transformers\s+import|AutoModel\w*|AutoTokenizer|huggingface|hugging\s+face|datasets\.load_dataset|load_dataset\(", 3, None),
    (r"import\s+xgboost|import\s+lightgbm|import\s+catboost|xgboost|lightgbm", 3, None),
    (r"import\s+numpy|numpy|np\.(?:array|dot|matmul|linalg|random|exp|mean|zeros|ones)", 1, None),
    (r"import\s+pandas|pandas|pd\.(?:DataFrame|read_csv|Series)|dataframe", 1, None),
    (r"matplotlib|seaborn|plt\.(?:plot|show|scatter|imshow)", 1, None),
    (r"import\s+cv2|opencv|PIL\.Image|torchvision|timm", 2, None),
    (r"\bnltk\b|\bspacy\b|gensim|sentencepiece|tiktoken", 2, None),
    (r"gym(?:nasium)?\.make|stable[_-]baselines", 3, None),
    (r"langchain|llama[_-]?index|openai\.(?:ChatCompletion|chat)|vllm", 2, None),
    (r"\bpython\b|def\s+\w+\(|```python", 0.5, None),
    # v0.2 expansion: training / LLM / RL / data-science libraries
    (r"from\s+peft|import\s+peft|LoraConfig|get_peft_model|from\s+trl|SFTTrainer|DPOTrainer|GRPOTrainer|PPOTrainer|"
     r"import\s+accelerate|from\s+accelerate|deepspeed|bitsandbytes|BitsAndBytesConfig|unsloth|flash_attn", 3, None),
    (r"from\s+diffusers|StableDiffusionPipeline|sentence_transformers|SentenceTransformer|import\s+faiss|faiss\.|chromadb|"
     r"from\s+pinecone|import\s+pinecone|weaviate|qdrant|llama_cpp|ollama|import\s+openai|from\s+openai|import\s+anthropic|from\s+anthropic", 3, None),
    (r"import\s+lightning|pytorch_lightning|LightningModule|torch_geometric|import\s+dgl|keras_nlp|keras_cv|timm\.create_model|"
     r"onnxruntime|tensorrt|torch\.compile|torch\.cuda|\.to\(['\"]cuda|model\.eval\(\)|model\.train\(\)|torch\.no_grad", 3, None),
    (r"import\s+optuna|optuna\.|import\s+mlflow|mlflow\.|import\s+wandb|wandb\.|ray\.tune|from\s+ray|rllib|stable_baselines3|"
     r"import\s+gymnasium|import\s+gym\b|env\.step\(|env\.reset\(", 3, None),
    (r"sklearn\.\w+|GridSearchCV|RandomizedSearchCV|cross_val_score|StandardScaler|MinMaxScaler|LabelEncoder|OneHotEncoder|"
     r"Pipeline\(|RandomForest\w+|GradientBoosting\w+|LogisticRegression|KMeans|DecisionTree\w+|SVC\(|accuracy_score|"
     r"confusion_matrix|classification_report|roc_auc_score|mean_squared_error", 3, None),
    (r"nn\.(?:Linear|Conv[123]d|LSTM|GRU|Embedding|Transformer\w*|MultiheadAttention|Dropout|BatchNorm\w*|LayerNorm|ReLU|"
     r"CrossEntropyLoss|MSELoss)|F\.(?:relu|softmax|cross_entropy|log_softmax)|optim\.(?:Adam\w*|SGD)|lr_scheduler|"
     r"tf\.(?:data|GradientTape|function)|layers\.(?:Dense|Conv2D|LSTM|Embedding|Dropout)", 3, None),
    (r"import\s+statsmodels|scipy\.optimize|scipy\.stats|import\s+networkx|import\s+polars|import\s+dask|pyspark\.ml", 1, None),
]

# Negative contexts: matched case-insensitively, each subtracts from the core score.
NEGATIVE_TERMS: list[tuple[str, float]] = [
    (r"artificial[\s-]+insemination", 6),
    (r"adobe[\s-]+illustrator|\.ai[\s-]+files?", 4),
    (r"cnn[\s-]+(?:news|anchor|reported?s?|reporter|politics|interview|journalist|host)", 4),
    (r"(?:power|electrical|step[\s-]?(?:up|down)|voltage|distribution)[\s-]+transformers?", 4),
    (r"optimus[\s-]+prime|autobots?|decepticons?|bumblebee", 5),
    (r"neural[\s-]+(?:tube|crest|plate)", 3),
    (r"(?:amnesty[\s-]+international)", 3),
    (r"ai[\s-]+weiwei", 4),
    (r"\d+\s?ml\b", 0),  # millilitres: only neutralises the 'ML' acronym, handled below
]

# Text-level patterns that hint at a topic without contributing to relevance.
TOPIC_PATTERNS: dict[str, str] = {
    "concepts": r"^\s*(?:what\s+(?:is|are|does)|define|explain|describe|how\s+does|why\s+(?:do|does|is)|difference\s+between)",
    "history": r"\b(?:history|invented|inventor|pioneer|founded|first\s+(?:ever|time)|origin(?:s|ally)?|in\s+(?:19[4-9]\d|20[0-2]\d))\b",
    "coding": r"```|\bdef\s+\w+\(|\bimport\s+\w+|\bclass\s+\w+\(|python\s+code|write\s+(?:a\s+)?(?:python\s+)?(?:function|script|program)",
    "theory_math": r"\$[^$]+\$|\\\(|\\\[|\\frac|\\sum|\\nabla|\\partial|\bderiv(?:e|ation)\b|\bprove\b|\btheorem\b",
    "research": r"\b(?:paper|papers|this\s+study|experiments?|our\s+(?:method|approach|model))\b",
    "ethics_society": r"\b(?:ethic(?:s|al)|privacy|fairness|discriminat\w*|bias(?:ed)?|responsib\w+|regulat\w+|surveillance|misinformation|jobs?)\b",
    "applications": r"\b(?:used\s+(?:in|for)|use\s+cases?|applications?\s+of|in\s+(?:healthcare|medicine|finance|education|agriculture|retail|manufacturing|transport\w*))\b",
}


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
            out.append(p[start:i])
            start = i + 1
        i += 1
    return out + [p[start:]]


def _compile(terms, flags, boundary=r"(?!\w)"):
    """A list of regexes per term: one per top-level alternative. Python's `re` has no trie
    optimisation, and a pattern that starts with an alternation loses the literal-prefix fast scan,
    so many small patterns are several times faster than one big alternation."""
    # No leading \b: it disables the literal-prefix fast scan. The left boundary is checked in _score.
    return [[re.compile(r"(?:" + a + r")" + boundary, flags) for a in _split_alts(p)] for p, *_ in terms]


_CORE_RE = _compile(CORE_TERMS, 0)          # matched on lower-cased text
_CORE_CS_RE = _compile(CORE_TERMS_CS, 0)    # matched on original text
_MATH_RE = _compile(MATH_TERMS, 0)          # lower-cased
_CODE_RE = _compile(CODE_TERMS, 0, boundary="")  # original text, no word boundary (code tokens)
_NEG_RE = [(re.compile(p), w) for p, w in NEGATIVE_TERMS if w]
_ML_UNIT_RE = re.compile(r"\d+\s?ml\b")
_TOPIC_RE = {k: re.compile(v, re.MULTILINE) for k, v in TOPIC_PATTERNS.items()}
MAX_CHARS = 4000  # per field; heads of long answers are representative enough


@dataclass
class KeywordResult:
    core: float = 0.0
    math: float = 0.0
    code: float = 0.0
    neg: float = 0.0
    hits: list[str] = field(default_factory=list)
    topic_hints: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> float:
        return max(self.core, self.math, self.code)

    def to_dict(self) -> dict:
        return {
            "core": round(self.core, 2), "math": round(self.math, 2), "code": round(self.code, 2),
            "neg": round(self.neg, 2), "hits": self.hits[:20],
            "topic_hints": {k: round(v, 2) for k, v in self.topic_hints.items()},
        }


def _score(regexes, terms, text, hits, topic_hints, skip=None, left_boundary=True):
    """Sum of weights over *distinct* terms, plus a small bonus for repetitions."""
    score = 0.0
    for idx, alts in enumerate(regexes):
        cnt = 0
        for rx in alts:
            for m in rx.finditer(text):
                st = m.start()
                if left_boundary and st and (text[st - 1].isalnum() or text[st - 1] in "_-"):
                    continue
                if skip and skip(m):
                    continue
                cnt += 1
                if cnt >= 5:
                    break
            if cnt >= 5:
                break
        if not cnt:
            continue
        pat, w, topic = terms[idx]
        s = w * (1.0 + 0.25 * (cnt - 1))
        score += s
        hits.append(pat if len(pat) <= 40 else pat[:37] + "...")
        if topic:
            topic_hints[topic] = topic_hints.get(topic, 0.0) + s
    return score


def score_text(text: str, weight: float = 1.0, result: KeywordResult | None = None) -> KeywordResult:
    """Score one piece of text and accumulate into `result` (scaled by `weight`)."""
    r = result or KeywordResult()
    if not text:
        return r
    text = text[:MAX_CHARS]
    low = text.lower()
    hits: list[str] = []
    th: dict[str, float] = {}

    has_ml_unit = bool(_ML_UNIT_RE.search(low))
    core = _score(_CORE_RE, CORE_TERMS, low, hits, th)
    core += _score(_CORE_CS_RE, CORE_TERMS_CS, text, hits, th,
                   skip=(lambda m: has_ml_unit and m.group(0) == "ML"))
    neg = sum(w for rx, w in _NEG_RE if rx.search(low))
    math_s = _score(_MATH_RE, MATH_TERMS, low, hits, th)
    code_s = _score(_CODE_RE, CODE_TERMS, text, hits, th, left_boundary=False)

    # Length normalisation: long answers accumulate weak hits; damp them gently.
    n_words = max(1, text.count(" ") + 1)
    damp = 1.0 if n_words < 300 else max(0.5, 1.0 / math.sqrt(n_words / 300))

    r.core += weight * max(0.0, core - neg) * damp
    r.math += weight * math_s * damp
    r.code += weight * code_s * damp
    r.neg += weight * neg
    r.hits.extend(h for h in hits if h not in r.hits)
    for k, v in th.items():
        r.topic_hints[k] = r.topic_hints.get(k, 0.0) + weight * v
    for k, rx in _TOPIC_RE.items():
        if rx.search(low):
            r.topic_hints[k] = r.topic_hints.get(k, 0.0) + weight * 1.0
    return r


def score_record(question: str, answer: str | None, q_weight: float = 1.0, a_weight: float = 0.5) -> KeywordResult:
    """Question is what the user asks -> weighted higher than the answer."""
    r = score_text(question, q_weight)
    if answer:
        score_text(answer, a_weight, r)
    return r


# Thresholds used by the splitter (and the collector pre-filter).
# score >= ACCEPT  -> AI without needing the model
# score <  REJECT  -> not AI, never sent to the model
# in between       -> sent to the zero-shot classifier
THRESHOLDS = {
    #          reject  accept
    "core": (2.0, 6.0),
    "math": (3.0, 99.0),   # math alone is never auto-accepted: "matrix" != AI. Zero-shot decides.
    "code": (3.0, 6.0),
}


def keyword_decision(kw: KeywordResult, domain: str = "ai_core") -> str:
    """Return 'accept' | 'reject' | 'uncertain'.

    `domain` is the relevance test of the source: math vocabulary only counts for
    math sources (otherwise every physics/finance chat with "probability" would pass).
    ML-library code vocabulary counts everywhere.
    """
    (rc, ac), (rm, _am), (rk, ak) = THRESHOLDS["core"], THRESHOLDS["math"], THRESHOLDS["code"]
    if kw.core >= ac or kw.code >= ak:
        return "accept"
    signal = kw.core >= rc or kw.code >= rk or (domain == "ai_math" and kw.math >= rm)
    return "uncertain" if signal else "reject"
