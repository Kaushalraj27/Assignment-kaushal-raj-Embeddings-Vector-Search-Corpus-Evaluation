"""
evaluation.py

- Expects folder `corpus/` inside the working directory with:
    speech1.txt ... speech6.txt

- Uses sentence-transformers to embed each speech.
- Computes cosine similarities and prints top-3 nearest other speeches for each file.
- Saves embeddings.npy and sentences.json for re-use.
"""

import os
import json
import numpy as np
from pathlib import Path

# use sentence-transformers directly (avoids langchain import issues)
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError(
        "sentence_transformers not available. Install with:\n"
        "    pip install -U sentence-transformers\n"
        f"Original error: {e}"
    )

def load_corpus(corpus_dir: str):
    p = Path(corpus_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}. Please create with the 6 speech*.txt files.")
    # collect files speech1..speech6 in sorted order
    files = sorted([f for f in p.iterdir() if f.is_file() and f.name.lower().startswith("speech") and f.suffix == ".txt"])
    if len(files) < 1:
        raise FileNotFoundError(f"No speech*.txt files found in {corpus_dir}.")
    texts = []
    names = []
    for f in files:
        names.append(f.name)
        with f.open("r", encoding="utf-8") as fh:
            texts.append(fh.read().strip())
    return names, texts

def compute_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    return embeddings

def cosine_sim_matrix(emb):
    # L2-normalize then dot product to get cosine similarities
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    emb_norm = emb / norms
    sim = emb_norm @ emb_norm.T
    return sim

def print_topk_neighbors(names, texts, embeddings, top_k=3):
    sim = cosine_sim_matrix(embeddings)
    n = len(names)
    print(f"\nEmbeddings shape: {embeddings.shape}\n")
    for i in range(n):
        # exclude self: set self-sim to -inf to avoid selecting itself
        row = sim[i].copy()
        row[i] = -np.inf
        idxs = np.argsort(row)[::-1]  # descending
        print(f"=== Query: {names[i]} ===")
        # show top_k neighbors
        for rank in range(min(top_k, len(idxs))):
            j = idxs[rank]
            score = float(row[j])
            snippet = texts[j][:200].replace("\n", " ")  # print small snippet
            print(f"  Rank {rank+1}: {names[j]}  (score={score:.4f})")
            print(f"    Snippet: {snippet!s}")
        print()

def save_outputs(names, embeddings, out_emb="embeddings.npy", out_json="sentences.json"):
    np.save(out_emb, embeddings)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"names": names}, fh, indent=2, ensure_ascii=False)
    print(f"Saved embeddings -> {out_emb}")
    print(f"Saved file list -> {out_json}")

def main():
    CORPUS_DIR = "corpus"
    # Load corpus
    names, texts = load_corpus(CORPUS_DIR)

    # Compute embeddings
    embeddings = compute_embeddings(texts)

    # Print top-k nearest speeches for each speech
    print_topk_neighbors(names, texts, embeddings, top_k=3)

    # Save outputs for the rest of your assignment (search scripts expect these files)
    save_outputs(names, embeddings)

if __name__ == "__main__":
    main()
