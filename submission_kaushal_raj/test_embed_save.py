# test_embed_save.py
from sentence_transformers import SentenceTransformer
import numpy as np
import json

def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    sentences = [
        "Hello world",
        "This is a test sentence for embeddings.",
        "How similar is this to hello world?"
    ]

    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    print("Embeddings shape:", embeddings.shape)

    # Save
    np.save("embeddings.npy", embeddings)          # numpy binary
    with open("sentences.json", "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)

    print("Saved embeddings.npy and sentences.json")

if __name__ == "__main__":
    main()
