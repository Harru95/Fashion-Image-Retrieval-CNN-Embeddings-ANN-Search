# Fashion Visual Search & Vibe Matcher

A production-ready fashion image retrieval system built with a fine-tuned ResNet18, Annoy ANN indexing, and a four-mode Gradio interface — deployed as a live Hugging Face Space.

**[Live Demo →](https://huggingface.co/spaces/Harsh9590/fashion-visual-search)**

---

## What it does

Given a fashion catalogue of ~5,000 items, the system retrieves visually similar products in real time using deep CNN embeddings and approximate nearest-neighbour search. Four independent search modes are supported:

| Mode | Description |
|---|---|
| **Similar by Index** | Enter any item ID to find visually similar pieces (category-aware filtering) |
| **Vibe Match** | Enter multiple item IDs (space-separated) to blend their embeddings into a single style query |
| **Text / Category Search** | Search by subcategory keyword (e.g. `dress`, `shoes`, `watch`) |
| **Image Upload** | Upload any fashion photo to find matching items in the catalogue |

---

## Architecture

```
Fine-tuned ResNet18 (FastAI)
        │
        ▼
512-dim penultimate-layer embeddings
        │
        ▼
Annoy index (Euclidean distance)  ←──  query vector
        │
        ▼
Top-k nearest neighbours → Gradio gallery
```

- **Backbone:** ResNet18 fine-tuned via FastAI on the fashion catalogue; **96.5% subcategory classification accuracy**
- **Embeddings:** 512-dimensional vectors extracted via a penultimate-layer forward hook (same layer used to build the index, guaranteeing consistency)
- **Index:** Annoy (Approximate Nearest Neighbours Oh Yeah) with Euclidean distance; sub-second retrieval
- **Category centroid caching:** Text search precomputes per-category mean embeddings at startup rather than per-request, reducing latency
- **Interface:** Gradio 4-tab UI deployed on Hugging Face Spaces

---

## Retrieval Evaluation

Evaluated across all 4,965 items using subCategory as ground truth:

| Metric | Score |
|---|---|
| Precision@6 | **0.91** |
| NDCG@6 | **0.91** |
| MRR | **0.95** |

26 out of 27 categories scored above 0.82. The single outlier (Free Gifts) scores lower due to visual heterogeneity — items in that category have no consistent visual signature.

---

## Stress Test

All four search modes were stress-tested across quality, edge case, and load tiers:

**81 / 81 tests passed — 0% error rate**

Key edge cases validated: boundary indices (0, max), empty inputs, invalid IDs, single-item vibe blend, non-existent category text, oversized n values, concurrent load.

---

## Repo Structure

```
├── fashion-retrieval.ipynb   # Full training pipeline: data prep, fine-tuning, embedding extraction, indexing
├── app.py                    # Gradio app (all four search modes)
├── requirements.txt          # Dependencies
├── emb_demo.npy              # 512-dim embeddings for ~5k items
├── df_demo.pkl               # Item metadata (paths, subcategory labels)
├── ann_demo.ann              # Annoy index
└── README.md
```

> **Note:** `fashion_resnet18.pkl` (the fine-tuned model, ~46MB) is hosted on the Hugging Face Space and not committed here to keep the repo lightweight.

---

## Setup

```bash
git clone https://github.com/Harru95/Fashion-Image-Retrieval-CNN-Embeddings-ANN-Search.git
cd Fashion-Image-Retrieval-CNN-Embeddings-ANN-Search
pip install -r requirements.txt
```

Download `fashion_resnet18.pkl` from the [HF Space files](https://huggingface.co/spaces/Harsh9590/fashion-visual-search/tree/main) and place it in the root directory, then:

```bash
python app.py
```

---

## Tech Stack

`Python` · `PyTorch` · `FastAI` · `ResNet18` · `Annoy` · `NumPy` · `Pandas` · `Gradio` · `Hugging Face Spaces` · `Docker`

---

## Key Engineering Decisions

**Why penultimate-layer embeddings?** The final classification layer collapses 512 dimensions to class logits, losing visual detail. The penultimate layer retains the full embedding and generalises better to unseen items and uploaded images.

**Why Annoy over exact search?** With 5k items, exact search is fast enough, but Annoy keeps retrieval sub-second even as the catalogue scales. The index is built once and loaded at startup.

**Why category-aware filtering in Similar by Index?** Raw ANN results occasionally surface cross-category neighbours (e.g. returning bags for a shoe query). Filtering to same-category neighbours first, then falling back to raw ANN if none exist, improves relevance without hurting recall.
