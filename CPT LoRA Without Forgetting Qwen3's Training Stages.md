# CPT LoRA Without Forgetting Qwen3's Training Stages

Oct 20, 2026 · @xyz

## What can be forgotten

The newest and smallest training stages are the easiest to forget, so a CPT LoRA puts long context, reasoning and post-training skills most at risk. Qwen3's stages, approximately:

| Stage | What it taught | Risk from domain CPT |
| --- | --- | --- |
| S1: general (\~30T tokens, 4K length) | General knowledge, language | Low (huge, deep in the weights) |
| S2: reasoning (\~5T tokens, STEM/code/math) | Reasoning, math, code | Medium |
| S3: long context (32K length, RoPE base 1M) | Long documents | High |
| Post-training (SFT + RL, or distillation for small models) | Chat, thinking mode, instruction following | Very high if CPT is done on the instruct model |

## 1. Make CPT look like the last stage

The model forgets less when the new training looks like what it saw last.

- **Sequence length:** train with long sequences (16K–32K), not 2K–4K. Pack short documents into one long sequence, and include some really long documents (full textbook chapters).
- **RoPE settings:** keep them the same (base 1,000,000). Do not change them.
- **Learning rate:** Qwen ends pretraining with a decayed, small LR. A big LR "re-warms" the model and causes forgetting. Use a small peak LR, short warmup, then decay to near zero.
- **Data mix:** the last stages were reasoning-heavy, so mix domain text with STEM, math and code data.

| Share | Data |
| --- | --- |
| 60–70% | Your domain text |
| 15–20% | Math / STEM / code (FineMath, OpenWebMath, The Stack) |
| 10–15% | General high-quality text (FineWeb-Edu) |
| Some | Long documents |

## 2. Use the model's own text as replay

Qwen's real training data is not public, so generate text with the original base model itself and mix it into CPT (5–10%). This text matches the model's own knowledge exactly, so it protects that knowledge very well.

Stronger option: add a **KL loss** to the original model on general data. The model is penalized when it moves away from the original on non-domain text.

## 3. Do CPT on Base, not on Instruct

Raw-text CPT on the instruct/thinking model damages chat, thinking mode and RL skills, and Qwen's post-training cannot be rebuilt in-house. Use this order instead:

1. Train the **CPT LoRA on Qwen3-8B-Base**.
2. Apply that same LoRA delta onto **Qwen3-8B (instruct)**. This works because Base and Instruct are very close in weight space (the "chat vector" idea).
3. Train the **SFT LoRA on Instruct + CPT LoRA**. The SFT step also repairs small chat damage.

Test step 2 carefully: it usually works, but not always perfectly.

In weights, step 2 is: W\_new = W\_instruct + (W\_base+CPT − W\_base) = W\_instruct + B·A. So you simply load the CPT LoRA, trained on Base, on top of Instruct. The closest published method is [Shadow-FT](https://arxiv.org/abs/2505.12716), which tested exactly this on Qwen 3; see References for the other papers.

## 4. Consider knowledge as Q&A instead of raw text

Raw-text CPT is the most dangerous step for forgetting. An alternative is to turn textbooks into many **rephrased explanations and question–answer pairs** (with a teacher model) and train only SFT.

The model also learns facts better when it sees them in many different forms, not as one textbook sentence. Many teams use a mix: small CPT plus large rephrased SFT.

## 5. Measure every stage before and after

Run the same tests on the original model and on your model; if any general score drops more than 1–2 points, increase replay, lower the LR, or reduce the LoRA rank (higher rank = more learning but also more forgetting).

| Ability (stage) | Test |
| --- | --- |
| General knowledge (S1) | MMLU, C-Eval |
| Reasoning (S2) | GSM8K, MATH, HumanEval |
| Long context (S3) | RULER or needle-in-haystack at 32K |
| Chat / thinking (post-training) | IFEval, MT-Bench, your own chat tests |
| Your domain | Your own exam set |

## 6. Serve the base model + LoRA adapters with vLLM

Load the base model once on the GPU and register each subject adapter by name; each request picks its adapter in the `model` field ([vLLM LoRA docs](https://docs.vllm.ai/en/stable/features/lora/)).

**Prepare each adapter.** One PEFT folder per subject (`adapter_config.json` + `adapter_model.safetensors`), already combined (CPT + SFT) into one adapter. Keep folders on local disk, so the server needs no internet. Do not train `embed_tokens` / `lm_head` in the adapters.

**Start the server:**

```bash
export HF_HUB_OFFLINE=1
vllm serve /models/Qwen3-8B \
  --served-model-name qwen3-8b \
  --enable-lora \
  --lora-modules \
    '{"name":"dl-tutor-v1","path":"/adapters/dl-tutor-v1","base_model_name":"qwen3-8b"}' \
    '{"name":"physics-tutor-v1","path":"/adapters/physics-tutor-v1","base_model_name":"qwen3-8b"}' \
    '{"name":"math-kids-v1","path":"/adapters/math-kids-v1","base_model_name":"qwen3-8b"}' \
  --max-lora-rank 320 \
  --max-loras 4 \
  --max-cpu-loras 16 \
  --max-model-len 32768
```

**Call an adapter** (OpenAI-compatible API; use `qwen3-8b` for the plain base):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "physics-tutor-v1",
       "messages": [{"role": "user", "content": "Explain Kirchhoff current law."}]}'
```

| Flag | What it does | How to set it |
| --- | --- | --- |
| `--enable-lora` | Turns on LoRA support | Always on |
| `--lora-modules` | Registers adapters at startup (name, path, base model) | One JSON entry per subject |
| `--max-lora-rank` | Largest adapter rank allowed; memory is reserved for it | Your largest rank (CPT 256 + SFT 32 = 288). Older versions accept only fixed values, so 320 is used here |
| `--max-loras` | Adapters active in the same GPU batch | Number of subjects used at the same time |
| `--max-cpu-loras` | Adapters cached in CPU RAM | All your subjects |
| `--lora-target-modules` | Limits which layers get LoRA | Leave unset (all layers) unless tuning speed |

**Production tips:**

- **Use static `--lora-modules` in production.** Runtime loading (`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, `/v1/load_lora_adapter`) is useful in testing, but vLLM warns it is a security risk outside an isolated, fully trusted environment.
- **Do not set `--max-lora-rank` too high.** A value far above your real rank wastes GPU memory and slows serving.
- **Router:** the app maps course → adapter name (for example "Electronic Physics" → `physics-tutor-v1`). A gateway such as LiteLLM or nginx can sit in front.
- **Versioning:** put the version in the adapter name (`-v1`, `-v2`). Keep the old adapter loaded until the new one passes your tests, so rollback is one name change.
- **Scaling:** run several identical vLLM replicas (same base, same adapters) behind a load balancer.
- **Benchmark before launch** with `vllm bench serve`: mixed-adapter traffic is a little slower than base-only traffic.

## Short plan

This order keeps most of what Qwen learned in all its stages and adds the domain on top.

1. Train the CPT LoRA on **Base**, with long sequences, reasoning replay, self-generated replay and a low LR.
2. Apply the CPT LoRA to **Instruct**.
3. Train the SFT LoRA on top.
4. Combine both into one adapter.
5. Test all stages against the original model.
6. Serve the base once with vLLM and register one adapter per subject.

## References

These papers support moving a weight delta trained on one model (Base) to its sibling (Instruct).

| Paper | What it shows | How it applies here |
| --- | --- | --- |
| [Shadow-FT: Tuning Instruct via Base](https://arxiv.org/abs/2505.12716) (Wu et al., 2025) | Paired Base and Instruct weights differ by less than 2% on average (Llama 3.1 8B). Fine-tune Base, then graft the weight update onto Instruct; beats tuning Instruct directly. Tested on Qwen 3 and Llama 3 ([code](https://github.com/wutaiqiang/Shadow-FT)) | Closest match to step 2: train the LoRA on Qwen3-Base, apply it to Qwen3-Instruct |
| [Chat Vector](https://arxiv.org/abs/2310.04799) (Huang et al., ACL 2024) | Chat vector = Llama2-chat − Llama2. Adding it to a continually pretrained model gives chat skills in a new language with no extra training ([code](https://github.com/aqweteddy/ChatVector)) | Reverse view of the same idea: CPT model + chat vector = Instruct + CPT delta |
| [ParamΔ](https://arxiv.org/abs/2504.21023) (Cao et al., ICLR 2025) | Model = Θ\_post − Θ\_base + Θ'\_base. Moving Llama3 post-training onto Llama3.1-base reached about 95% of Llama3.1-Instruct, with no training | Post-training transfers to an updated base, so it also survives your CPT update |
| [Fine-tuning Transfer](https://arxiv.org/abs/2503.20110) (Lin et al., 2025) | Diff vectors transfer across model versions; works best when the models are linearly connected in weight space. The merged model is also a better start for further fine-tuning | Supports step 3: SFT on top of Instruct + CPT delta |
| [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) (Ilharco et al., ICLR 2023) | Task vector = fine-tuned − pretrained weights; task vectors can be added and negated. Requires the same pretrained starting point | The basic theory behind all methods above |
| [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (Biderman et al., TMLR 2024) | LoRA learns less than full fine-tuning in CPT (gap not closed even at high rank) but forgets less of the source domain | Why a CPT LoRA is safer, and why high rank is needed |
| [vLLM LoRA Adapters docs](https://docs.vllm.ai/en/stable/features/lora/) | Serving flags, request format, runtime loading warning | Section 6 |
