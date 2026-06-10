---
title: Fashion Visual Retrieval
emoji: 👗
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 6.15.2
app_file: app.py
pinned: false
license: mit
---

# Fashion Visual Retrieval & Vibe-Matching

Find visually similar fashion items using deep image embeddings.

- A **ResNet18** (fine-tuned via FastAI, **96.5%** subcategory-classification accuracy) produces
  **512-dimensional embeddings** from its penultimate layer.
- Embeddings are indexed with **Annoy** (Euclidean distance) for sub-second similarity search.
- Three modes: **Similar items** (nearest neighbours, category-aware), **Vibe match** (blend several
  items' embeddings into one style query), and **Text/category** search.

Demo runs on a stratified subset of the catalogue. Full project + evaluation:
see the GitHub repo.