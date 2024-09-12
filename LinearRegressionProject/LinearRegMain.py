import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the CSV
data = pd.read_csv('NFLX.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Normalize the 'Volume' and 'Close' data
scaler = StandardScaler()
data[['Volume', 'Close']] = scaler.fit_transform(data[['Volume', 'Close']])

# Plot 'Date' vs 'Close' price
plt.figure(figsize=(10, 6))
plt.scatter(data['Date'], data['Close'], color='blue', label='Closing Price')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Netflix Stock Closing Prices Over Time')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.legend()
plt.show()

# Define the loss function (for linear regression)
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].Volume  # Using 'Volume' as x
        y = points.iloc[i].Close   # Using 'Close' as y
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

# Define the gradient descent function
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].Volume  # Using 'Volume' as x
        y = points.iloc[i].Close   # Using 'Close' as y
        
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    
    # Update m and b
    m_now = m_now - (m_gradient * L)
    b_now = b_now - (b_gradient * L)
    
    return m_now, b_now

# Initialize parameters
m = 0  # Slope
b = 0  # Intercept
L = 0.00001  # Increased learning rate for better convergence
epochs = 300

# Gradient descent process
for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}, Loss: {loss_function(m, b, data)}")
    m, b = gradient_descent(m, b, data, L)

print(f"Final values - Slope (m): {m}, Intercept (b): {b}")

# Plot the regression line along with the data points
plt.scatter(data['Volume'], data['Close'], color="black", label='Data Points')
plt.plot(data['Volume'], m * data['Volume'] + b, color='red', label='Regression Line')
plt.xlabel('Volume')
plt.ylabel('Close Price')
plt.legend()
plt.show()
