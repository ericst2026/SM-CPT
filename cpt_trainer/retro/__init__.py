"""RETRO-fit + LoRA continued pre-training of Qwen3 (see docs/retro_fit_lora_cpt.md).

Pipeline (each step is resumable and writes into its own folder):
    db.py          AI documents -> retrieval database (Qwen3 tokens, 64-token chunks, doc keys / text hashes)
    data.py        1 B-token training mix (same corpora / proportions / val split as the web5b run) packed into
                   16-24 k document-masked sequences, with the 64-token query chunks of every sequence
    embed.py       frozen sentence encoder (bge-small-en-v1.5) for chunk embeddings
    index.py       faiss IVF-PQ index over the database chunk embeddings (sharded, resumable)
    neighbours.py  k nearest database chunks for every training / validation chunk, with leakage filters
    model.py       neighbour encoder + chunked cross-attention (CCA) layers on the frozen decoder
    train.py       joint training of the RETRO layers and a LoRA adapter (PEFT, modules_to_save="retro")
    eval.py        MMLU / ai_large / MCQ with online retrieval, validation loss with / without / random neighbours
"""
