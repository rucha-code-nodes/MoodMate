# # # app.py
# # from flask import Flask, jsonify, render_template
# # import cv2
# # import tensorflow as tf
# # import numpy as np

# # app = Flask(__name__)

# # # Load model and classifier
# # model = tf.keras.models.load_model('emotion_recognition_model.keras')
# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# # # Movie mapping
# # movie_map = {
# #     "happy": ["Zindagi Na Milegi Dobara", "The Pursuit of Happyness", "Paddington"],
# #     "sad": ["Inside Out", "The Fault in Our Stars", "Taare Zameen Par"],
# #     "angry": ["Gladiator", "John Wick", "Mad Max: Fury Road"],
# #     "surprise": ["Inception", "Shutter Island", "The Prestige"],
# #     "fear": ["The Conjuring", "A Quiet Place", "Get Out"],
# #     "disgust": ["Joker", "Parasite", "Nightcrawler"],
# #     "neutral": ["Forrest Gump", "The Social Network", "The King's Speech"]
# # }

# # @app.route("/")
# # def index():
# #     return render_template("index.html")

# # @app.route("/detect", methods=["GET"])
# # def detect_emotion():
# #     cap = cv2.VideoCapture(0)
# #     ret, frame = cap.read()
# #     cap.release()

# #     if not ret:
# #         return jsonify({"emotion": "none", "movies": []})

# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# #     if len(faces) == 0:
# #         return jsonify({"emotion": "none", "movies": []})

# #     x, y, w, h = faces[0]
# #     face = frame[y:y+h, x:x+w]
# #     face = cv2.resize(face, (224, 224)) / 255.0
# #     face = np.expand_dims(face, axis=0)

# #     pred = model.predict(face)
# #     emotion = classes[np.argmax(pred)]

# #     return jsonify({"emotion": emotion, "movies": movie_map.get(emotion, [])})

# # if __name__ == "__main__":
# #     app.run(debug=True)


# # app.py
# from flask import Flask, jsonify, render_template
# import cv2
# import tensorflow as tf
# import numpy as np

# app = Flask(__name__)

# # Load model and classifier
# model = tf.keras.models.load_model('emotion_recognition_model.keras')
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# # Movie mapping
# movie_map = {
#     "happy": ["Zindagi Na Milegi Dobara", "The Pursuit of Happyness", "Paddington"],
#     "sad": ["Inside Out", "The Fault in Our Stars", "Taare Zameen Par"],
#     "angry": ["Gladiator", "John Wick", "Mad Max: Fury Road"],
#     "surprise": ["Inception", "Shutter Island", "The Prestige"],
#     "fear": ["The Conjuring", "A Quiet Place", "Get Out"],
#     "disgust": ["Joker", "Parasite", "Nightcrawler"],
#     "neutral": ["Forrest Gump", "The Social Network", "The King's Speech"]
# }

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/detect", methods=["GET"])
# def detect_emotion():
#     cap = cv2.VideoCapture(0)
#     detected_emotion = "none"

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#         for (x, y, w, h) in faces:
#             roi_color = frame[y:y + h, x:x + w]
#             face = cv2.resize(roi_color, (224, 224)) / 255.0
#             face = np.expand_dims(face, axis=0)

#             pred = model.predict(face)
#             emotion = classes[np.argmax(pred)]
#             detected_emotion = emotion

#             # Draw rectangle and label
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 255, 0), 2, cv2.LINE_AA)

#         cv2.imshow("Emotion Detection", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to break
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     return jsonify({"emotion": detected_emotion, "movies": movie_map.get(detected_emotion, [])})

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64

app = Flask(__name__)
model = load_model('emotion_recognition_model.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    data = request.get_json()
    img_data = data['image'].split(',')[1]
    img_array = np.frombuffer(base64.b64decode(img_data), dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'emotion': 'No face detected', 'movies': []})

    (x, y, w, h) = faces[0]
    roi = frame[y:y+h, x:x+w]
    roi = cv2.resize(roi, (224, 224))
    roi = roi.astype("float") / 255.0
    roi = np.expand_dims(roi, axis=0)  # shape now: (1, 224, 224, 3)

    prediction = model.predict(roi)
    max_index = int(np.argmax(prediction))
    emotion = emotion_labels[max_index]

    movie_suggestions = {
        'Happy': ['Zindagi Na Milegi Dobara', '3 Idiots', 'Barfi!'],
        'Sad': ['Tamasha', 'Kal Ho Naa Ho', 'Taare Zameen Par'],
        'Angry': ['Gangs of Wasseypur', 'Singham', 'Rang De Basanti'],
        'Fear': ['Bhoot', 'Stree', 'Pari'],
        'Surprise': ['Kahaani', 'Drishyam', 'Special 26'],
        'Neutral': ['Swades', 'October', 'Lunchbox'],
        'Disgust': ['Article 15', 'Padman', 'Pink']
    }

    return jsonify({
        'emotion': emotion,
        'movies': movie_suggestions.get(emotion, [])
    })

if __name__ == '__main__':
    app.run(debug=True)
