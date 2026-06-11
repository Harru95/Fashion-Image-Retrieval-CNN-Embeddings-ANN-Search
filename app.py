"""
app.py  --  Fashion Visual Retrieval (Hugging Face Space, Gradio)
Four working modes, including REAL image-upload search via the fine-tuned ResNet18.
Requires: emb_demo.npy, df_demo.pkl, ann_demo.ann, images/, fashion_resnet18.pkl
"""

import numpy as np
import pandas as pd
import gradio as gr
import torch
import os
from annoy import AnnoyIndex
from collections import defaultdict
from fastai.vision.all import load_learner
from torchvision import transforms

EMB_DIM = 512

# ---------- load artifacts once at startup ----------
embeddings = np.load("emb_demo.npy")
df = pd.read_pickle("df_demo.pkl").reset_index(drop=True)
ann = AnnoyIndex(EMB_DIM, "euclidean")
ann.load("ann_demo.ann")

paths  = df["img_path"].tolist()
labels = df["subCategory"].tolist()
CATEGORIES = sorted(set(labels))

cat_to_idx = defaultdict(list)
for i, c in enumerate(labels):
    cat_to_idx[str(c).lower()].append(i)

cat_centroids = {
    cat: np.mean([embeddings[i] for i in idxs[:50]], axis=0).astype("float32")
    for cat, idxs in cat_to_idx.items()
}

# ---------- load model + register the SAME penultimate-layer hook as the notebook ----------
learn = load_learner("fashion_resnet18.pkl")
learn.model.eval()

_penultimate = learn.model[1][-2]          # same layer used to build the indexed embeddings
_captured = {}

def _hook(module, inp, out):
    _captured["feat"] = out.detach()

_penultimate.register_forward_hook(_hook)
_transform=transforms.Compose([transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),])

def embed_image(pil_img):
    """Embed a NEW image exactly like the indexed items were embedded."""
    img = pil_img.convert("RGB")
    tensor=_transform(img).unsqueeze(0)
    _captured.clear()
    with torch.no_grad():
        learn.model(tensor)             # triggers a forward pass -> hook fires
    feat = _captured["feat"]
    if feat.ndim > 2:                       # (B, C, H, W) -> global average pool -> (B, C)
        feat = feat.mean(dim=[2, 3])
    return feat.squeeze().cpu().numpy().astype("float32")

import os

def _gallery(indices):
    results = []
    for i in indices:
        filename = os.path.basename(paths[i])  # extracts just "12345.jpg"
        space_path = f"images/{filename}"       # maps Space's images/ folder
        results.append((space_path, f"ID: {i} | {labels[i]}"))
    return results

# ---------- search functions ----------
def similar_by_index(idx, n=6):
    n=int(n)
    try:
        idx = int(idx)
        if not (0 <= idx < len(paths)):
            return []
        nbrs = [j for j in ann.get_nns_by_item(idx, n * 6 + 1) if j != idx]
        same = [j for j in nbrs if labels[j] == labels[idx]] or nbrs
        return _gallery([idx]) + _gallery(same[:n])
    except Exception:
        return []

def vibe_match(tokens_str, n=6):
    n=int(n)
    try:
        parts = [int(x) for x in tokens_str.replace(",", " ").split() if x.strip().isdigit()]
        valid = [p for p in parts if 0 <= p < len(paths)]
        if not valid:
            return []
        vibe_vec = np.mean([embeddings[p] for p in valid], axis=0)
        nbrs = [x for x in ann.get_nns_by_vector(vibe_vec, n + len(valid)) if x not in valid]
        return _gallery(nbrs[:n])
    except Exception:
        return []

def text_search(query, n=6):
    n=int(n)
    q = str(query).strip().lower()
    if not q:
        return []
    if q in cat_centroids:
        center=cat_centroids[q]
    else:
        match = [i for cat, idxs in cat_to_idx.items() if q in cat for i in idxs]
        if not match:
            return []
        center = np.mean([embeddings[i] for i in match[:50]], axis=0)
    return _gallery(ann.get_nns_by_vector(center, n))

def recom_from_upload(pil_img, n=6):
    n=int(n)
    if pil_img is None:
        return []
    try:
        vec = embed_image(pil_img)
        return _gallery(ann.get_nns_by_vector(vec, n))
    except Exception as e:
        print("upload error:", e)
        return []

# ---------- UI ----------
with gr.Blocks(title="Fashion Visual Matcher") as demo:
    gr.Markdown("# 👗 Fashion Visual Search & Vibe Matcher")
    gr.Markdown("Visual recommendations via a fine-tuned ResNet18 (96.5% subcategory accuracy) + Annoy nearest-neighbour search.")

    with gr.Tab("Similar items"):
        gr.Markdown("Enter the numerical ID of any item to find visually similar pieces.")
        idx_in = gr.Number(label=f"Item Index ID (0 to {len(paths)-1})", value=0, precision=0)
        n1 = gr.Slider(2, 12, value=6, step=1, label="How many results")
        out1 = gr.Gallery(label="Most similar items", columns=6, height="auto")
        gr.Button("Find Similar").click(similar_by_index, [idx_in, n1], out1)

    with gr.Tab("Vibe match"):
        gr.Markdown("Enter several item IDs (space-separated) to blend their styles.")
        idx_multi = gr.Textbox(label="Item indices (e.g. 0 10 45)", value="0 1 2")
        n2 = gr.Slider(2, 12, value=6, step=1, label="How many results")
        out2 = gr.Gallery(label="Items matching the blended vibe", columns=6, height="auto")
        gr.Button("Blend & Search").click(vibe_match, [idx_multi, n2], out2)

    with gr.Tab("Text / category"):
        gr.Markdown(f"Search by category. Try: **{', '.join(CATEGORIES[:6])}**")
        q_in = gr.Textbox(label="Search text category")
        n3 = gr.Slider(2, 12, value=6, step=1, label="How many results")
        out3 = gr.Gallery(label="Results", columns=6, height="auto")
        gr.Button("Search Category").click(text_search, [q_in, n3], out3)

    with gr.Tab("Image Upload Search"):
        gr.Markdown("Upload any fashion photo to find visually similar items in the catalogue.")
        upload_in = gr.Image(type="pil", label="Upload fashion image")
        n4 = gr.Slider(2, 12, value=6, step=1, label="How many results")
        out4 = gr.Gallery(label="Visually matching items", columns=6, height="auto")
        gr.Button("Find Matches from Upload").click(recom_from_upload, [upload_in, n4], out4)

if __name__ == "__main__":
    demo.launch(show_error=True)