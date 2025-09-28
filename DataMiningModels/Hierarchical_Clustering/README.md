# Hierarchical Clustering – Mall Customers

This project applies **Hierarchical Clustering** to segment customers from the **Mall Customers dataset**.  
The goal is to identify distinct groups of customers based on their **Annual Income** and **Spending Score**, which can help in targeted marketing strategies.

---

## 📂 Project Structure

- **`Mall_Customers.csv`** – Dataset containing customer demographic and spending information  
- **`script.py`** – Python implementation of hierarchical clustering and visualization  

---

## 🚀 Features

- **Dendrogram Analysis**  
  - Uses **Ward’s method** to compute the linkage  
  - Helps determine the **optimal number of clusters**  

- **Agglomerative Clustering**  
  - Clusters customers based on **Annual Income** and **Spending Score**  
  - Uses **cosine distance** and **complete linkage**  

- **Visualization**  
  - Scatter plot of clusters with distinct colors  
  - Labeled clusters to show different customer segments  

---

## 🔧 Requirements

Install dependencies before running:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
