# Neural Network for MNIST Digit Classification from Scratch  

### A Neural Network Model for Handwritten Digit Recognition  

This project involves building a neural network model from scratch to classify handwritten digits from the MNIST dataset with a high degree of accuracy. The model features a **fully connected two-layer architecture**, utilizing **ReLU activation** for hidden layers and **Softmax** for the output layer.  

This implementation focuses on fundamental neural network concepts and techniques without relying on pre-built machine learning frameworks, making it an excellent demonstration of low-level deep learning principles.  

---

## Features  

- **Custom neural network model** developed from scratch without pre-built frameworks  
- **Two-layer fully connected architecture** for accurate digit classification  
- **ReLU activation** for hidden layers and **Softmax activation** for output  
- **Forward and backward propagation** implemented manually  
- **Gradient descent optimization** with learning rate decay for fine-tuning  
- **Training progress visualized** with accuracy and loss plots over iterations  
- **Model evaluation** using confusion matrix and misclassified image analysis  

This project highlights the application of neural network fundamentals, providing a clear understanding of core techniques like backpropagation, optimization, and evaluation.  

---

## Tools  

This project uses the following tools and libraries:  

- **Python** - Programming language used for implementation  
- **NumPy** - For numerical operations and matrix calculations  
- **pandas** - For handling and analyzing data  
- **Matplotlib** - For plotting accuracy and loss trends  
- **Seaborn** - For advanced visualizations like confusion matrix heatmaps  
- **Scikit-Learn** - For evaluation metrics and dataset preprocessing  

---

## Running the Program  

To run the neural network model:  

1. Ensure the required libraries are installed using `pip install numpy pandas matplotlib seaborn scikit-learn`.  
2. Download the MNIST dataset and ensure it is in the proper directory.  
3. Execute the Python script:  

```bash
python3 mnist_neural_network.py
