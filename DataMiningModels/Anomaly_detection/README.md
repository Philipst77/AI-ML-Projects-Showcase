# Novelty Detection with Local Outlier Factor (LOF)

This project demonstrates **novelty detection** using the **Local Outlier Factor (LOF)** algorithm from `scikit-learn`.  
The goal is to distinguish between normal data points and novel outliers that were not part of the training set.

---

## 📂 Project Structure

- **`script.py`** – Main Python script implementing LOF novelty detection and visualization  

---

## 🚀 Features

- Uses **Local Outlier Factor (LOF)** with `novelty=True` for novelty detection  
- Trains on synthetic 2D Gaussian clusters  
- Tests on both **new normal data** and **abnormal outliers**  
- Visualizes:
  - Learned decision boundary (frontier)
  - Training samples
  - Novel regular points
  - Novel abnormal outliers  

---
