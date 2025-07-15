import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))

# Create a synthetic anomalous image
anomalous_image = np.random.rand(28 * 28)

# Build the AutoEncoder model
model = tf.keras.Sequential([
    # Encoder: Reduce dimensionality, learn the most important features
    tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)), # Reducing dimension to 128
    tf.keras.layers.Dense(64, activation='relu'), # Further reducing dimension to 64
    tf.keras.layers.Dense(32, activation='relu'), # Further reducing to the most compact form (bottleneck layer)

    # Decoder: Reconstruct the image from the reduced representation
    tf.keras.layers.Dense(64, activation='relu'), # Start expanding dimension
    tf.keras.layers.Dense(128, activation='relu'), # Continue expanding dimension
    tf.keras.layers.Dense(x_train.shape[1], activation='sigmoid') # Restore to original image size
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(x_train, x_train, epochs=20, batch_size=256, validation_data=(x_test, x_test))

# Function to calculate reconstruction loss
def calculate_reconstruction_loss(data, model):
    reconstructions = model.predict(data)
    reconstruction_errors = np.mean(np.abs(data - reconstructions), axis=1)
    return reconstruction_errors

# Evaluate the model
reconstruction_loss_normal = calculate_reconstruction_loss(x_test, model)
reconstruction_loss_anomalous = calculate_reconstruction_loss(np.array([anomalous_image]), model)

# Print average reconstruction loss
print(f"Average Reconstruction Loss for Normal Data: {np.mean(reconstruction_loss_normal)}")
print(f"Reconstruction Loss for Anomalous Data: {reconstruction_loss_anomalous[0]}")

# Visualization of reconstruction error distribution
plt.figure(figsize=(6, 4))
plt.hist(reconstruction_loss_normal, bins=50, alpha=0.6, color='g', label='Normal')
plt.axvline(x=reconstruction_loss_anomalous[0], color='r', linestyle='dashed', linewidth=2, label='Anomalous')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()