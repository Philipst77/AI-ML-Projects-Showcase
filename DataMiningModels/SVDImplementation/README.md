# 🍷 Dimensionality Reduction with SVD and Logistic Regression

This project demonstrates how to apply **Singular Value Decomposition (SVD)** for dimensionality reduction, followed by **logistic regression** to classify wine samples. The objective is to compare model performance before and after dimensionality reduction to assess whether reducing dimensionality improves efficiency without compromising accuracy.

---

## 🔧 Setup

Ensure Python 3 is installed. Then install the necessary libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## 📁 Files

- `svd_logistic.py` – Applies SVD to reduce the dataset to 2 dimensions and trains a logistic regression model.
- `original_logistic.py` – Trains a logistic regression model on the full dataset without dimensionality reduction.
- `wine.csv` – The Wine dataset (from UCI Machine Learning Repository).
- `results.md` – Summary of accuracy scores and evaluation between both models.

---

## 🚀 How to Run

Run the model using dimensionality reduction:

```bash
python3 svd_logistic.py
```

Run the model using the original dataset:

```bash
python3 original_logistic.py
```

Review accuracy and evaluation metrics in `results.md`.

---

## 🎯 Objectives

- Practice using **SVD** to reduce high-dimensional data to fewer features.
- Train and evaluate models using **logistic regression**.
- Compare model performance **with and without dimensionality reduction**.
- Understand the trade-offs between **model complexity and accuracy**.

---

## 📊 Outcome

This exercise highlights when dimensionality reduction is worthwhile—particularly when working with datasets where feature redundancy or high dimensionality may affect performance. It also emphasizes how reduced models can maintain strong classification accuracy while improving efficiency and interpretability.

---

## 📎 Dataset Reference

- [Wine Dataset – UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Wine)

