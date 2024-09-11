<<<<<<< HEAD
import threading
import cv2
from deepface import DeepFace

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Set video capture width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# Load the reference image
reference_img = cv2.imread("ref1.jpg")

# Function to check if the face matches
def check_face(frame):
    global face_match
    try:
        # Use DeepFace to verify the face match with a specific model and distance metric
        if DeepFace.verify(frame, reference_img.copy(), model_name='Facenet', distance_metric='cosine', threshold=0.4)['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False

# Main loop to read frames and check for a face match
while True:
    ret, frame = cap.read()  # Capture a frame from the webcam
    if ret:
        if counter % 38 == 8:  # Run face check periodically
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()  # Start the face check in a new thread
            except ValueError:
                pass
        counter += 1

        # Display result on the video feed
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 225), 3)

        cv2.imshow("video", frame)  # Show the video feed with the result

    # Exit the loop if the 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
=======
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
data = pd.read_csv('NFLX.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

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
L = 0.000001  # Learning rate (adjusted for stock data)
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
>>>>>>> 85d695b ( LinearRegressionProj)
