# K-Means and Kernel K-Means Clustering

This project implements both **standard K-Means** and **Kernel K-Means** clustering algorithms.  
It demonstrates the difference between linear and nonlinear clustering approaches using sample datasets.

---

## 📂 Project Structure

- **`standard_kMeansClustering.py`** – Implementation of the standard K-Means clustering algorithm  
- **`kernel_kMeansClustering.py`** – Implementation of Kernel K-Means (using nonlinear kernels for clustering)  
- **`test1_data.txt`**, **`test2_data.txt`** – Example datasets to run clustering experiments  

---

## 🚀 Features

- **Standard K-Means**  
  - Iteratively assigns points to nearest cluster centroid  
  - Updates centroids until convergence  

- **Kernel K-Means**  
  - Uses kernel functions (e.g., polynomial, RBF)  
  - Captures **nonlinear cluster boundaries**  

- **Dataset Support**  
  - Works on synthetic datasets (`test1_data.txt`, `test2_data.txt`)  
  - Can be adapted for other datasets  

- **Visualization** (if included in scripts)  
  - Plots cluster assignments and boundaries  

---

## 🔧 Requirements

Install dependencies:

```bash
pip install numpy matplotlib
