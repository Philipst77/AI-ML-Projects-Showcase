# Autoencoder for Anomaly Detection (MNIST)

This project demonstrates the use of an **Autoencoder** for **anomaly detection**.  
The autoencoder is trained on the **MNIST dataset** to learn compressed representations of handwritten digits.  
An anomalous image is introduced, and the model is evaluated based on **reconstruction error**.

---

## 📂 Project Structure

- **`script.py`** – Main implementation of the autoencoder for anomaly detection  
- **MNIST dataset** – Automatically downloaded via `tensorflow.keras.datasets`  
- **Synthetic anomalous image** – Random noise image used to test anomaly detection  

---

## 🚀 Features

- Builds a **deep autoencoder** with TensorFlow/Keras:
  - Encoder compresses MNIST digits into a bottleneck latent space  
  - Decoder reconstructs the original digit images  
- Calculates **reconstruction loss** for:
  - Normal MNIST test samples  
  - A synthetic anomalous sample  
- Compares distributions of reconstruction errors to highlight anomalies  
- Provides visualizations:
  - **Reconstruction error histogram** (normal vs anomalous)  
  - **Training vs validation loss curves**  

---

## 🔧 Requirements

Install dependencies before running:

```bash
pip install numpy matplotlib scikit-learn tensorflow
