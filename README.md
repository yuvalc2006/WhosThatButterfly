# 🦋 Who's That Butterfly?

**Authors**: Yuval Cohen & Inbar Rotshtein

## 📖 Introduction

"Who's That Butterfly?" is a deep learning project aimed at accurately identifying butterfly species from images using computer vision techniques. This project was developed as part of a university assignment and explores how advanced vision models like CLIP can be adapted for ecological research.

Butterflies are not just visually captivating—they are vital to ecosystems as pollinators and environmental indicators. Manual identification, however, is time-consuming and requires expertise. This project automates that task using machine learning.

---

## 📚 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Usage (in Colab)](#usage-in-colab)
- [Examples](#examples)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

---

## ✅ Features

- Automatically identifies butterfly species from images.
- Uses fine-tuned **OpenCLIP ViT-B-32** model as a feature extractor.
- Supports traditional classifiers: **SVM, KNN, Logistic Regression, MLP**.
- Performs **data preprocessing**, **dimensionality reduction (PCA, t-SNE)**, and **visualizations**.
- Tracks model performance via **accuracy**, **F1-score**, and **confusion matrix**.
- Built for **Google Colab** and integrates with **Google Drive** for dataset storage.

---

## 🚀 Usage (in Colab)

> ⚠️ This project is intended to be executed in **Google Colab**.

### Steps:

1. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Install Dependencies**:
   ```bash
   !pip install open_clip_torch scikit-image
   ```

3. **Set Paths**:
   Ensure your dataset is located in:
   ```
   /content/drive/MyDrive/ButterflyProject/
   ├── Training_set.csv
   └── train/
   ```

4. **Run Notebook**:
   Execute each notebook cell sequentially to load data, preprocess, train, and evaluate models.

---

## 🧪 Examples

### PCA Visualization
```python
pca2 = PCA(n_components=2)
plot_2d(pca2.fit_transform(X_train_std), y_train, "PCA (2D) — train")
```

### t-SNE Visualization
```python
tsne2 = TSNE(n_components=2, random_state=RANDOM_STATE)
plot_2d(tsne2.fit_transform(X_train_std[perm]), y_train[perm], "t-SNE (2D)")
```

---

## 📈 Results

- Achieved **93.2% accuracy** using **SVM with fine-tuned CLIP embeddings**.
- Fine-tuning improved feature representations, leading to:
  - Tighter intra-class clustering
  - Larger inter-class margins
  - Enhanced focus on butterfly-specific features (e.g., wing shapes and patterns)

---

## 🔍 Future Work

To further enhance performance:
- Increase input resolution and apply stronger augmentations (e.g., MixUp, CutMix)
- Use class balancing and background randomization
- Improve CLIP fine-tuning via better prompt design and head architectures
- Explore test-time augmentation and model ensembles
- Use Grad-CAM to interpret model focus

---

## 👥 Contributors

- Yuval Cohen
- Inbar Rotshtein

---

## 🔒 License

This project was developed as part of a university course. It is **not licensed for public reuse or redistribution**. Please do not copy, distribute, or modify the content without permission from the authors.

---
