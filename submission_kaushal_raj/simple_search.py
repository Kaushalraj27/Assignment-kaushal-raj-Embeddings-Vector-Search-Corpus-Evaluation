# simple_search.py
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import numpy.linalg as la

def cosine_sim(a, b):
    return (a @ b) / (la.norm(a) * la.norm(b))

def main():
    query = "Hi there"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([query], convert_to_numpy=True)[0]

    emb = np.load("embeddings.npy")
    sents = json.load(open("sentences.json"))

    scores = [cosine_sim(q_emb, e) for e in emb]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    for idx, score in ranked:
        print(f"{score:.4f}  {sents[idx]}")

if __name__ == "__main__":
    main()
