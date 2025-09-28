# Neural Networks on MNIST with L2, Dropout, and Early Stopping

This project explores training **fully-connected neural networks (MLPs)** on the **MNIST dataset**.  
The focus is on comparing different **regularization techniques** to reduce overfitting and improve generalization.  

---

## 📂 Project Structure

- **Data Files**
  - `MNISTXtrain1.npy`, `MNIST_y_train_1.npy` – Training data and labels  
  - `MNIST_X_test_1.npy`, `MNIST_y_test_1.npy` – Validation/test data  
  - `MNIST_autolabTest_X.npy` – Hidden test set for evaluation  

- **Models**
  - `m1.py` – Baseline NN (no regularization)  
  - `m2.py` – NN with **L2 weight regularization**  
  - `m3.py` – NN with **Dropout regularization**  
  - `m4.py` – NN with **Early Stopping**  

- **Utilities**
  - `ConvertToOddEven.py` – Converts digit classification (0–9) into binary odd/even labels  
  - `barPlotTemplate.py` – Helper functions for bar plot visualizations  
  - `confusionMatrixHeatmap.py` – Utility to draw confusion matrix heatmaps  
  - `pa2pre*.py` – Data preprocessing helpers  

- **Outputs (per model)**
  - Loss curve plots (`m*_loss_curve.png`)  
  - Confusion matrix (`m*_confusion_matrix.png`)  
  - Accuracy per class bar charts (`m*_accuracy_per_class.png`)  
  - Predictions (`m*_predictions.txt`, `m*_final_predictions.txt`)  

---

## 🚀 Features

- Implements and compares **4-layer neural networks** under different setups:
  - **Model 1:** No regularization (baseline)  
  - **Model 2:** L2 weight regularization  
  - **Model 3:** Dropout layers (prevent co-adaptation)  
  - **Model 4:** Early stopping on validation loss  

- Evaluates models using:
  - **Training & validation loss curves**  
  - **Confusion matrices**  
  - **Per-class accuracy breakdowns**  
  - **Odd vs Even classification results**  

---

## 🔧 Requirements

Install dependencies:

```bash
pip install numpy tensorflow keras matplotlib seaborn scikit-learn
