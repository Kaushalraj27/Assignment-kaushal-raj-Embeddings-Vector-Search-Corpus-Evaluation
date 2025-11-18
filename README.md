# Assignment-kaushal-raj-Embeddings-Vector-Search-Corpus-Evaluation

# Assignment – Embeddings, Vector Search & Corpus Evaluation

This repository contains the full implementation for:
- Generating sentence embeddings  
- Saving embeddings  
- Vector similarity search  
- Evaluating retrieval performance over a 6-document corpus  

---

##  Project Structure

```
assignment/
│
├── test_embed.py
├── test_embed_save.py
├── simple_search.py
├── evaluation.py
├── create_corpus.py
│
├── embeddings.npy
├── sentences.json
│
├── corpus/
│   ├── speech1.txt
│   ├── speech2.txt
│   ├── speech3.txt
│   ├── speech4.txt
│   ├── speech5.txt
│   ├── speech6.txt
│
├── requirements.txt
└── README.md
```

---

##  1. Virtual Environment Setup

Run these commands in PowerShell:

```powershell
python -m venv venv_final
.env_final\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

##  2. Running the Scripts

### **A) Generate Embeddings**
```powershell
python test_embed.py
```

### **B) Save Embeddings**
```powershell
python test_embed_save.py
```

### **C) Vector Similarity Search**
```powershell
python simple_search.py
```

### **D) Create 6-Document Corpus**
```powershell
python create_corpus.py
```

### **E) Evaluate Retrieval on Corpus**
```powershell
python evaluation.py
```

---

##  3. Input Overview
- Sample sentences for Task 1–3  
- Six long-form documents inside `corpus/`  

---

## 4. Output Overview
- Embedding matrix (N × 384)  
- `embeddings.npy`  
- `sentences.json`  
- Top-k cosine similarity results  
- Top-3 similar document retrieval per speech  

---

##  5. Submission ZIP Contents

```
test_embed.py
test_embed_save.py
simple_search.py
evaluation.py
create_corpus.py

embeddings.npy
sentences.json

corpus/
    speech1.txt
    speech2.txt
    speech3.txt
    speech4.txt
    speech5.txt
    speech6.txt

requirements.txt
README.md
```
