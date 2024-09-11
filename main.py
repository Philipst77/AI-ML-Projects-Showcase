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
