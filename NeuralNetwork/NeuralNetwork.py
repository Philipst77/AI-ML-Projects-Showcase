import numpy as np  # for linear algebra and working with matrices
import pandas as pd  # for reading data
import matplotlib.pyplot as plt  # for plotting graphs
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load data (Make sure to specify the correct path and filename)
# Load the MNIST dataset from a CSV file. Adjust the path to point to your local file.
mnist_data = pd.read_csv('mnist_test.csv')  # Use the correct path to the CSV file

# Convert DataFrame to a numpy array for faster computation
mnist_data = np.array(mnist_data)
num_rows, num_columns = mnist_data.shape

# Shuffle the data to ensure a random distribution of samples
np.random.shuffle(mnist_data)

# Split the data into development and training sets
# First 1000 samples for validation (dev set), remaining for training
dev_data = mnist_data[0:1000].T  # Transpose to match the required input format
y_dev = dev_data[0]  # First row contains labels
x_dev = dev_data[1:num_columns] / 255.0  # Normalize pixel values to [0, 1]

train_data = mnist_data[1000:num_rows].T
y_train = train_data[0]  # First row contains labels
x_train = train_data[1:num_columns] / 255.0  # Normalize pixel values to [0, 1]
_, num_train_samples = x_train.shape

# Initialize the weights and biases for the neural network
def initialize_parameters():
    # Initialize weights with small random values and biases with a small negative offset
    weights_1 = np.random.rand(10, 784) - 0.5  # 10 neurons, 784 inputs
    bias_1 = np.random.rand(10, 1) - 0.5  # 10 neurons, 1 bias each
    weights_2 = np.random.rand(10, 10) - 0.5  # 10 neurons in the output layer, 10 inputs
    bias_2 = np.random.rand(10, 1) - 0.5  # 10 output neurons
    return weights_1, bias_1, weights_2, bias_2

# ReLU activation function
# ReLU is used to introduce non-linearity by setting negative values to zero
def relu_activation(z):
    return np.maximum(0, z)

# Softmax activation function for multi-class classification
# Converts raw scores into probabilities by normalizing exponentiated values
def softmax_activation(z):
    exp_values = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return exp_values / np.sum(exp_values, axis=0)

# Forward propagation step
# Computes the activations for the hidden layer and output layer
def forward_propagation(weights_1, bias_1, weights_2, bias_2, input_data):
    z1 = weights_1.dot(input_data) + bias_1  # Linear combination for hidden layer
    a1 = relu_activation(z1)  # Apply ReLU activation
    z2 = weights_2.dot(a1) + bias_2  # Linear combination for output layer
    a2 = softmax_activation(z2)  # Apply Softmax activation for output layer
    return z1, a1, z2, a2

# Derivative of ReLU for backpropagation
def relu_derivative(z):
    return z > 0  # Returns a binary array: 1 if z > 0, else 0

# Convert labels to one-hot encoded format for multi-class classification
def one_hot_encode(labels):
    one_hot_matrix = np.zeros((labels.size, labels.max() + 1))
    one_hot_matrix[np.arange(labels.size), labels] = 1
    return one_hot_matrix.T

# Backward propagation step
# Computes gradients for weights and biases to update them during training
def backward_propagation(z1, a1, z2, a2, weights_1, weights_2, input_data, labels):
    num_samples = labels.size
    one_hot_labels = one_hot_encode(labels)
    dz2 = a2 - one_hot_labels  # Gradient of loss w.r.t output layer activation
    dw2 = 1 / num_samples * dz2.dot(a1.T)  # Gradient of loss w.r.t weights_2
    db2 = 1 / num_samples * np.sum(dz2, axis=1, keepdims=True)  # Gradient of loss w.r.t bias_2
    dz1 = weights_2.T.dot(dz2) * relu_derivative(z1)  # Gradient of loss w.r.t hidden layer
    dw1 = 1 / num_samples * dz1.dot(input_data.T)  # Gradient of loss w.r.t weights_1
    db1 = 1 / num_samples * np.sum(dz1, axis=1, keepdims=True)  # Gradient of loss w.r.t bias_1
    return dw1, db1, dw2, db2

# Update weights and biases using gradient descent with learning rate decay
def update_parameters(weights_1, bias_1, weights_2, bias_2, dw1, db1, dw2, db2, learning_rate, iteration):
    # Apply learning rate decay over iterations to fine-tune convergence
    decayed_learning_rate = learning_rate * (1.0 / (1.0 + 0.01 * iteration))
    weights_1 -= decayed_learning_rate * dw1
    bias_1 -= decayed_learning_rate * db1
    weights_2 -= decayed_learning_rate * dw2
    bias_2 -= decayed_learning_rate * db2
    return weights_1, bias_1, weights_2, bias_2

# Predict the class with the highest probability for each sample
def make_predictions(activations_output):
    return np.argmax(activations_output, axis=0)

# Calculate the accuracy of predictions
def calculate_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size

# Gradient descent optimization
def gradient_descent(training_data, training_labels, num_iterations, learning_rate):
    weights_1, bias_1, weights_2, bias_2 = initialize_parameters()
    accuracy_history = []  # List to store accuracy at each step
    
    for i in range(num_iterations):
        z1, a1, z2, a2 = forward_propagation(weights_1, bias_1, weights_2, bias_2, training_data)
        dw1, db1, dw2, db2 = backward_propagation(z1, a1, z2, a2, weights_1, weights_2, training_data, training_labels)
        weights_1, bias_1, weights_2, bias_2 = update_parameters(weights_1, bias_1, weights_2, bias_2, dw1, db1, dw2, db2, learning_rate, i)
        
        if i % 10 == 0:
            predictions = make_predictions(a2)
            accuracy = calculate_accuracy(predictions, training_labels)
            accuracy_history.append(accuracy)  # Store accuracy for plotting
            print("Iteration:", i)
            print("Accuracy:", accuracy)
    
    return weights_1, bias_1, weights_2, bias_2, accuracy_history

# Visualize misclassified images
def plot_misclassified_images(data, labels, predictions, num_examples=5):
    misclassified_indices = np.where(predictions != labels)[0]
    plt.figure(figsize=(10, 5))

    for i in range(min(num_examples, len(misclassified_indices))):
        index = misclassified_indices[i]
        image = data[:, index].reshape((28, 28)) * 255
        plt.subplot(1, num_examples, i+1)
        plt.title(f'Pred: {predictions[index]}, Actual: {labels[index]}')
        plt.imshow(image, cmap='gray')
        plt.axis('off')

    plt.show()

# Run gradient descent to train the model
final_weights_1, final_bias_1, final_weights_2, final_bias_2, accuracy_history = gradient_descent(x_train, y_train, 500, 0.10)

# Plot accuracy and loss over iterations
loss_history = [2 - acc for acc in accuracy_history]  # Example loss calculation

plt.figure(figsize=(10, 5))
plt.plot(accuracy_history, label='Accuracy', color='blue')
plt.plot(loss_history, label='Loss', color='orange')
plt.xlabel('Iterations (per 10)')
plt.ylabel('Value')
plt.title('Training Accuracy and Loss over Iterations')
plt.legend()
plt.grid(True)
plt.show()

# Make predictions on the validation set
validation_predictions = make_predictions(forward_propagation(final_weights_1, final_bias_1, final_weights_2, final_bias_2, x_dev)[3])

# Plot misclassified images
plot_misclassified_images(x_dev, y_dev, validation_predictions)

# Generate and plot the confusion matrix
confusion_mat = confusion_matrix(y_dev, validation_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
