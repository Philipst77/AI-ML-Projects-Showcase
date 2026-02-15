# Machine Learning & Artificial Intelligence Projects

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A curated collection of machine learning and artificial intelligence implementations covering classical statistical models, modern deep learning architectures, and applied real-world systems. Each subdirectory represents a self-contained project focused on algorithmic foundations, experimental analysis, and performance evaluation.

---

## Overview

This repository consolidates projects spanning supervised learning, unsupervised learning, optimization, anomaly detection, neural networks, and applied computer vision. The implementations emphasize conceptual clarity, mathematical grounding, and from-scratch development where appropriate.

The collection serves as both a learning framework and a demonstration of practical model implementation across multiple domains.

---

## Repository Structure

### DataMiningModels

A comprehensive set of classical and modern machine learning algorithms implemented for experimentation and analytical study.

**Core Areas Covered**

- Association Rule Mining (Apriori)
- Anomaly Detection (Local Outlier Factor, Autoencoders)
- Artificial Neural Networks with dropout and early stopping
- Dimensionality Reduction (SVD)
- Linear and Regularized Regression (L1/L2)
- Support Vector Machines
- K-Nearest Neighbors with Cross Validation
- Kernel K-Means
- Hierarchical Clustering
- Latent Dirichlet Allocation (Topic Modeling)
- Imbalanced Dataset Classification
- Bias–Variance Analysis and OLS/WLS Regression

The implementations include visual diagnostics such as confusion matrices, dendrograms, clustering heatmaps, and regression plots.

---

### FacialRecognitionProject

A real-time facial recognition system integrating deep learning–based face embeddings with live video processing.

**Core Components**

- Real-time webcam-based inference
- DeepFace / FaceNet embedding extraction
- Cosine similarity–based identity verification
- Adjustable decision thresholds
- Multithreaded frame processing for improved throughput

**Technology Stack**

- Python
- OpenCV
- DeepFace / TensorFlow / Keras
- Multithreading for performance optimization

---

### LinearRegressionProject – Netflix Stock Modeling

A regression-based modeling project exploring predictive relationships between trading volume and closing price for Netflix stock.

**Implemented Approaches**

- Closed-form Ordinary Least Squares solution
- Custom Gradient Descent optimization
- Learning rate and epoch experimentation
- Regression visualization and error analysis

**Tools**

- Python
- pandas
- matplotlib

This project is intended strictly for educational experimentation.

---

### NeuralNetwork – MNIST Classification from Scratch

A two-layer fully connected neural network implemented from first principles to classify handwritten digits from the MNIST dataset.

**Implementation Details**

- ReLU activation for hidden layers
- Softmax output layer
- Manual forward propagation
- Manual backpropagation
- Gradient descent with learning rate scheduling
- Per-class accuracy metrics and confusion matrix visualization

No high-level deep learning frameworks are used for training logic; core operations are implemented using NumPy.

---

## Design Principles

- Emphasis on algorithmic transparency
- Manual implementation where feasible
- Clear experimental evaluation
- Modular project separation
- Strong focus on mathematical understanding

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Philipst77/ML-AI-Projects.git
cd ML-AI-Projects
```

Each subdirectory contains project-specific instructions and dependencies.
