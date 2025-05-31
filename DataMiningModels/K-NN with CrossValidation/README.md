# 📝 Yelp Review Sentiment Classification using K-Nearest Neighbors (KNN)

This project implements a custom **K-Nearest Neighbor classifier** to predict sentiment from text-based Yelp restaurant reviews. The sentiment values are binary:  
- `+1` for positive sentiment  
- `-1` for negative sentiment  

The classifier was built from scratch (no `sklearn` KNN used) and fine-tuned through **cross-validation**, focusing on:
- Text preprocessing and feature engineering
- Different similarity metrics (cosine, Jaccard, etc.)
- Optimal value of **k**

---

## 📁 Dataset

Three files are included:
- `new_train.csv` — contains 18,000 labeled reviews (sentiment + text)
- `new_test.csv` — 18,000 reviews (text only), to be predicted
- `format.csv` — shows proper format for final predictions (18000 rows alternating +1/-1)

Each row in `new_train.csv` follows the format:



---


## 🧠 Key Steps

1. **Text Vectorization:**
   - Used `TfidfVectorizer` from scikit-learn
   - Limited to `max_features=1000` for performance

2. **Model:**
   - Custom implementation of KNN using cosine similarity
   - Used `numpy` for efficient vectorized computation
   - Precomputed vector norms to avoid repeated calculations
   - Selected best `k` using 10-fold cross-validation on training data

3. **Evaluation:**
   - Predictions generated for 18,000 test examples
   - Output saved in `my_predictions.csv` (matching `format.csv`)


---

## 🧪 Experiments

📈 Included in the `Report/` directory:
- Accuracy scores vs. `k` values
- Cosine vs. Jaccard performance comparison
- Vectorization scheme impact (TF-IDF vs BoW)

---

## ✅ Output

Final predictions are stored in `predictions.csv` and follow the format of `format.csv` (i.e., 18,000 rows, each row is either `+1` or `-1`).

---

## 🚀 Future Work

- Try approximate nearest neighbor (ANN) methods for scalability
- Integrate word embeddings (e.g., Word2Vec, GloVe) for better semantic representation
- Experiment with dimensionality reduction (e.g., PCA or Truncated SVD)

---

