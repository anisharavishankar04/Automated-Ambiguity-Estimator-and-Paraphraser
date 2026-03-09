# Automated-Ambiguity-Estimator-and-Paraphraser

An NLP pipeline that detects **ambiguous examination questions** and **rephrases them while preserving intent** using Large Language Models.

The system extracts questions from PDFs, assigns an **ambiguity score (0–100)** with explanation, and automatically rephrases unclear questions using a **dual-encoder validation pipeline**.

---

## Features

- PDF question extraction (supports scanned PDFs via OCR)
- Explainable ambiguity scoring
- Automatic question rephrasing
- Dual validation pipeline
  - Bi-encoder for global similarity
  - Cross-encoder for token-level similarity
- Dynamic temperature scaling for iterative rephrasing
- Human-in-the-loop fallback for uncertain cases

---

## Pipeline


---

## Models Used

| Task | Model |
|-----|------|
| Question Extraction | Llama-3.1-8B |
| Ambiguity Scoring | Llama-3.3-70B |
| Rephrasing | Llama-3.1-8B |
| Bi-Encoder | sentence-transformers/all-MiniLM-L6-v2 |
| Cross-Encoder | cross-encoder/ms-marco-MiniLM-L-6-v2 |

All LLM inference is performed via the **Groq API**.

---

## Results

Tested on **225 engineering and science questions**.

| Status | Percentage |
|------|------|
| Original Kept | 52.0% |
| Rephrased | 47.1% |
| Manual Review Required | 0.9% |

---

## Tech Stack

- Python
- FastAPI
- Groq API
- Sentence Transformers
- Tesseract OCR
- HTML / CSS
- Hugging Face Spaces

---

## Future Work

- Support for **tables, diagrams, and mathematical notation**
- Post-rephrasing ambiguity rescoring
- Larger dataset evaluation
