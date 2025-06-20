# SVM Classification with Social Network Ads Dataset

## Overview

This project demonstrates how to implement and evaluate a Support Vector Machine (SVM) classifier on a real-world dataset (`Social_Network_Ads.csv`). The dataset includes features like Age and Estimated Salary to predict whether a user purchased a product or not.

## Tasks Completed

- **Loaded and preprocessed data** using pandas and scikit-learn.
- **Split the dataset** into training and testing sets (75% training, 25% testing).
- **Applied feature scaling** to normalize the input features.
- **Trained two SVM classifiers** using:
  - Linear kernel
  - RBF (Radial Basis Function) kernel
- **Evaluated both models** using:
  - Predictions on the test set
  - Confusion matrix
  - Accuracy calculation
- **Visualized decision boundaries** on both training and testing data.

## Results

- With the **linear kernel**, the model achieved an accuracy of approximately **90%**.
- After switching to the **RBF kernel**, the accuracy improved to **93%**.
- The confusion matrix with the RBF kernel:

[[64 4]
[ 3 29]]


- **64** true negatives
- **29** true positives
- **4** false positives
- **3** false negatives

This suggests the RBF kernel is more effective for this classification task, likely due to its ability to handle non-linear decision boundaries better than the linear kernel.

## Dependencies

- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn

## How to Run

1. Clone this repository or download the `script.py` and the dataset `Social_Network_Ads.csv`.
2. Make sure all dependencies are installed (`pip install -r requirements.txt` if using a virtual environment).
3. Run the script:

 ```bash
 python3 script.py
