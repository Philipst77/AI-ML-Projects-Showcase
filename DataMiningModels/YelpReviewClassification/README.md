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

1. **Preprocessing:**
   - Lowercasing
   - Punctuation removal
   - Tokenization
   - Stopword removal
   - TF-IDF vectorization

2. **Model:**
   - Custom implementation of KNN
   - Experimented with various distance metrics:
     - Cosine similarity
     - Euclidean distance (on dense vectors)
     - Jaccard similarity (on binary bag-of-words)
   - Optimized `k` value using 5-fold cross-validation

3. **Evaluation:**
   - Accuracy metric
   - Validation on hold-out data
   - Generated prediction file for test set matching `format.csv`

---

## 🧪 Experiments

📈 Included in the `Report/` directory:
- Accuracy scores vs. `k` values
- Cosine vs. Jaccard performance comparison
- Vectorization scheme impact (TF-IDF vs BoW)

---

## 📦 Structure

```
├── src/
│ ├── knn.py # Custom KNN implementation
│ ├── preprocess.py # Text preprocessing pipeline
│ ├── vectorize.py # TF-IDF and BoW functions
│ ├── train.py # Model training and validation
│ ├── predict.py # Generate predictions for test set
│ └── utils.py # Helper functions
│
├── Report/
│ └── report.pdf # Final write-up with approach, results, and insights
│
├── new_train.csv
├── new_test.csv
├── format.csv
├── predictions.csv # Output predictions for test set
└── README.md

```







---

## ✅ Output

Final predictions are stored in `predictions.csv` and follow the format of `format.csv` (i.e., 18,000 rows, each row is either `+1` or `-1`).

---

## 🚀 Future Work

- Try approximate nearest neighbor (ANN) methods for scalability
- Integrate word embeddings (e.g., Word2Vec, GloVe) for better semantic representation
- Experiment with dimensionality reduction (e.g., PCA or Truncated SVD)

---

