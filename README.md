# 🧥 AI Fashion Recommendation System  
**Image Similarity • Text Search • Upload Search • Vibe Matching**  
Built with **FastAI, ResNet18, Annoy, PIL, Gradio**

---

## 🚀 Overview

This project is an **AI-powered Fashion Recommendation System** that finds visually similar fashion products using deep learning and vector similarity search.

The system supports **four powerful recommendation modes**:

1. **Search by Index** — Select any dataset image index to view similar items  
2. **Search by Text** — Search via subcategory names (e.g., "shoes", "bags")  
3. **Search by Image Upload** — Upload your own image and get similar recommendations  
4. **Vibe Match** — Enter multiple indices or keywords and get items that match the combined “vibe”

This project uses:

- **FastAI + ResNet18** for feature extraction  
- **Annoy (Approximate Nearest Neighbor)** for fast vector search  
- **PIL** for image handling  
- **Gradio** for a clean interactive UI  

---

## 📦 Features

### ✅ **1. Image-Based Similarity Search**
Extract embeddings from the penultimate layer of ResNet18 and retrieve similar images via Annoy.

### ✅ **2. Text-Based Search**
Enter text such as `"shirts"`, `"jeans"`, `"watches"` to retrieve related items.

### ✅ **3. Image Upload Search**
Upload any image → embedding is computed → similar fashion items are retrieved.

### ✅ **4. Vibe Matcher**
A unique mode where users can input:

- **Indices:** `10, 25, 87`  
- **Keywords:** `shoes, blue shirt, backpack`

The system averages the embeddings → finds items that represent the *combined vibe*.

---

## 📁 Project Structure

Project1/
│── fashion_rec.ipynb # Main notebook (training + ANN + UI)
│── app/ # (optional) UI code
│── models/ # saved models / ann index (optional)
│── data/ # dataset folder (ignored by git)
│── requirements.txt # dependencies
│── README.md # this file
│── LICENSE # MIT license

---

## 📥 Dataset

We use the **Fashion Product Images (Small)** dataset:

📌 Kaggle:  
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

Dataset structure after extraction:

fashion_data/images/<id>.jpg
fashion_data/styles.csv

---

## 🧠 Model & Embeddings

- **Model:** ResNet18 (FastAI)  
- **Embedding Dimension:** 512  
- **ANN Index:** Annoy (Euclidean distance)  
- **Trees:** 10  

Embedding extraction example:

```python
penultimate = learn.model[1][-2]
hook = FeatureHook(penultimate)
Saved vector index:

fashion_annoy.ann
🖥️ How to Run (Local or Colab)
🔹 1. Clone the Repository
git clone https://github.com/Harru95/Project1
cd Project1
🔹 2. Install Dependencies
pip install -r requirements.txt
🔹 3. Run the Notebook
Open fashion_rec.ipynb in:

Google Colab

Jupyter Notebook

VS Code Notebook

Execute all cells.

🔹 4. Launch the Gradio App
python
demo.launch()
You will get a link such as:
Running on public URL: https://xxxx.gradio.live
🧪 Sample Inputs for Testing
Index Search
Index: 50
Recommendations: 5
Text Search
shoes
tshirts
bags
Vibe Match (Indices)
10, 25, 87
Vibe Match (Text)
shoes, blue, backpack
formal shirt, watch
Image Upload
Upload any fashion image (shirt, shoe, watch, etc).

📝 License
This project is released under the MIT License.
You are free to use, modify, and distribute it.

🙌 Author
Harsh Soni

GitHub: https://github.com/Harru95

LinkedIn: https://linkedin.com/in/harsh-soni-7819522a41

⭐ Support
If you found this project helpful, please consider giving it a ⭐ on GitHub!
