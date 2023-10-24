# Import necessary libraries
import cv2  # OpenCV for video and image processing
import numpy as np  # NumPy for numerical operations
from keras.models import model_from_json  # Load a Keras model from a JSON file

# Dictionary mapping emotion labels to their corresponding names
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the emotion detection model from JSON and load its weights
json_file = open('model/emotion_model.json', 'r')  # Read the JSON file containing the model architecture
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)  # Create the model from the loaded JSON
emotion_model.load_weights("model/emotion_model.h5")  # Load the model weights
print("Loaded model from disk")

# Start the webcam feed (you can also specify a video file path instead of 0 for webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam feed
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))  # Resize the frame for better visualization
    if not ret:
        break

    # Create a face detector using a Haar Cascade Classifier
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in num_faces:
        # Draw a bounding box around the detected face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

        # Extract the region of interest (ROI) containing the face
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotion by checking the percentage of emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        # Display the predicted emotion label on the frame
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with emotion predictions
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam feed and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
