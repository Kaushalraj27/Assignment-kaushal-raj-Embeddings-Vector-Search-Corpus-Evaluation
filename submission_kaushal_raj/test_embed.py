# test_embed.py
from sentence_transformers import SentenceTransformer
import numpy as np

def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print("Loading model:", model_name)
    model = SentenceTransformer(model_name)   # will download model first run

    sentences = [
        "Hello world",
        "This is a test sentence for embeddings.",
        "How similar is this to hello world?"
    ]

    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
    print("Embeddings shape:", embeddings.shape)   # (N, dim)
    print("First embedding (first 10 values):", embeddings[0][:10].tolist())

if __name__ == "__main__":
    main()
