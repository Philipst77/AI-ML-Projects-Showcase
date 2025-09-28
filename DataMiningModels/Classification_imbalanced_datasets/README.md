# Classification on Imbalanced Banking Data

This project explores **classification on an imbalanced dataset** using the **Bank Marketing dataset**.  
The dataset contains information about bank clients and whether they subscribed to a term deposit after a marketing campaign.  

The goal is to preprocess the data, handle categorical variables, and prepare it for classification tasks such as **logistic regression**.

---

## 📂 Project Structure

- **`banking.csv`** – Raw dataset (Bank Marketing dataset)  
- **`Bankingdata.py`**, **Bankingdata1.py–Bankingdata4.py** – Python scripts for different stages of preprocessing and modeling  
- **`count_plot.png`** – Visualization of target variable distribution  

---

## 🚀 Features

- Loads and cleans the **Bank Marketing dataset**  
- Handles missing values and consolidates categorical features (e.g., collapsing multiple "basic" education categories into one)  
- Explores the **class imbalance problem**:
  - Term deposit subscription is **highly imbalanced** (most clients did not subscribe)  
  - Prints percentages of subscription vs. non-subscription  
- Uses **one-hot encoding** for categorical variables:
  - Features like `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`  
- Prepares a final dataset (`X`, `y`) ready for training classifiers such as **Logistic Regression**  

---

## 🔧 Requirements

Install the necessary dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
