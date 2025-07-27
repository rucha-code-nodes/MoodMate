
import cv2
import numpy as np
import tensorflow as tf

# Define classes
Classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load the trained model
model_path = 'D:\\Journey To Speech Emotion Recognition\\1_Emotion Detection\\emotion_recognition_model.keras'
try:
    new_model = tf.keras.models.load_model(model_path)
except Exception as e:
    raise IOError(f"Cannot load the model: {e}")

# Load face detector
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(face_cascade_path)
if faceCascade.empty():
    raise IOError("Cannot load face cascade classifier")

# Open webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        cv2.putText(frame, "No faces detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Choose only the first detected face
        (x, y, w, h) = faces[0]
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        face_roi = roi_color
        
        final_image = cv2.resize(face_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0

        Predictions = new_model.predict(final_image)
        
        emotion_index = np.argmax(Predictions)
        status = Classes[emotion_index]

        # Draw black background rectangle
        cv2.rectangle(frame, (x, y - 40), (x + 175, y), (0, 0, 0), -1)
        # Add text in yellow color
        cv2.putText(frame, status, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Draw face outline in blue
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Emotion Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








