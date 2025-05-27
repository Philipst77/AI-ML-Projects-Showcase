# Know Your Data: Exploratory Data Analysis with Python

This repository contains a hands-on walkthrough of **exploratory data analysis (EDA)** techniques applied to real-world health and anthropometric datasets. The goal is to understand the structure, distributions, and relationships within the data using statistics and visualization.

Two datasets are used:
- **Diabetes dataset**: Medical measurements used for diabetes prediction.
- **Height-Weight dataset**: Human height and weight data for distribution and correlation analysis.

---

## 🧠 Data-Driven Approach

This project simulates a practical EDA workflow:
1. **Data Loading & Structure Inspection**  
   Explore the shape, data types, and summary statistics.

2. **Univariate & Bivariate Visualization**  
   Generate scatter plots and histograms to visually examine distributions.

3. **Correlation Analysis**  
   Identify relationships between variables using correlation heatmaps.

4. **Multivariate Distribution Comparison**  
   Use KDE and ChainConsumer to analyze joint distributions by group.

5. **Insights & Reporting**  
   Summarize key patterns and relationships discovered during exploration.

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `data_exploration.py` | Basic data inspection: shape, types, missing values, and summary stats. |
| `scatter.py` | Scatter plots to analyze bivariate relationships. |
| `correlation.py` | Pearson correlation heatmap to highlight strong dependencies. |
| `histograms.py` | Height vs. weight histogram colored by gender. |
| `probabilityDist.py` | 2D KDE plots comparing male vs. female distributions. |
| `results.txt` | Summary of analysis and findings. |
| `diabetes.csv`, `height_weight.csv` | Datasets used in the analysis. |

---

## 📦 Setup

Make sure Python 3 is installed and run:

```bash
pip install pandas matplotlib seaborn chainconsumer

```
---
🚀 How to Run

-- python3 data_exploration.py
python3 scatter.py
python3 correlation.py
python3 histograms.py
python3 probabilityDist.py

---
🎯 Project Goals
Practice data cleaning and structural exploration

Visualize and understand relationships between variables

Use statistical tools to quantify correlation and overlap

Build a structured process for investigating raw data


