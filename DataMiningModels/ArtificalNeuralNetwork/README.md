# 🧠 Customer Churn Prediction with Artificial Neural Network (ANN)

This project implements a machine learning model using an Artificial Neural Network (ANN) to predict customer churn in a banking environment. By analyzing customer attributes—such as demographics, financial metrics, and account activity—the model identifies patterns that indicate whether a customer is likely to leave the bank.

---

## 📌 What This Project Does

This end-to-end pipeline includes:

- 🔄 **Data Loading & Preprocessing**  
  Reads a structured dataset of bank customers, performs data cleaning, encodes categorical variables (e.g., Gender, Geography), and scales numerical features to prepare the data for training.

- 🧠 **Model Architecture & Training**  
  Defines a deep neural network using Keras with multiple dense layers, optimized for binary classification (churn vs no churn). The model is trained on a labeled dataset with a train/test split to evaluate generalization.

- 📊 **Performance Evaluation**  
  Assesses the model’s performance on unseen data using accuracy, loss, and prediction metrics. You can optionally generate a confusion matrix or classification report for further insight.

- 🔮 **Churn Prediction for a New Customer**  
  After training, the model is used to make a real-time prediction on a new customer based on a provided feature set.

---

## 🚀 Quick Start

### 1. Install Python & Dependencies

Ensure you have Python 3.x installed. Then install all required packages:

```bash
pip install numpy pandas scikit-learn tensorflow
