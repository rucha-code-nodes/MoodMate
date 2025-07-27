Link: https://drive.google.com/file/d/1KdNTEOj1Awj8WrMvtxP50mAbxNDwvmym/view?usp=sharing

# 🎭 Emotion-Based Movie Recommender System 🎬

This project is an **AI-powered emotion recognition system** that detects human emotions using facial expressions via webcam and recommends suitable movies based on the detected emotion.

---

## 💡 Why I Built This Project

Emotions play a major role in how we consume entertainment. I wanted to build a system that could:

* Detect real-time emotions using facial expressions
* Recommend emotionally relevant movies
* Blend **Computer Vision**, **Deep Learning**, and **Flask Web App** to create a seamless and fun experience

This solution can be extended for **mental health support**, **personalized recommendations**, or **interactive entertainment systems**.

---

## 🔧 What I Used

### ✅ Tech Stack:

* **Python** for data processing and model development
* **TensorFlow & Keras** for deep learning (MobileNetV3 transfer learning)
* **OpenCV** for real-time face detection and preprocessing
* **Flask** for building the web backend and serving predictions
* **HTML/CSS/JavaScript** for the frontend
* **Haar Cascade** for face detection

### ✅ Dataset:

* Facial expression image dataset with 7 emotion categories:

  * Angry
  * Disgust
  * Fear
  * Happy
  * Neutral
  * Sad
  * Surprise

---

## 📁 Project Structure


├── run.py                  # Trains and builds the deep learning model
├── test.py                 # Real-time webcam emotion prediction (standalone)
├── app.py                  # Flask app for webcam-based emotion detection and movie recommendation
├── see.py                  # Basic webcam test to check setup
├── emotion_recognition_model.keras # Saved trained model
├── static/ and templates/  # HTML/CSS frontend files
└── haarcascade_frontalface_default.xml # Face detection model

## 🚀 How to Run

1. **Install dependencies**:

   
   pip install -r requirements.txt
  

2. **Run the Flask App**:

  
   python app.py


3. **Open your browser** and go to `http://127.0.0.1:5000/`

4. **Allow webcam access**, and it will detect your emotion and suggest movies accordingly!

---

## 🧠 Model Summary

* **Architecture**: MobileNetV3 (pretrained on ImageNet)
* **Custom Layers**: Added Dense layers for classification into 7 emotions
* **Training**: Data augmentation with `ImageDataGenerator`, model trained for 20–25 epochs
* **Accuracy**: \~95% on validation set

---

## 🎞️ Example Use Case

Detected Emotion → 🎉 **Happy**
Recommended Movies → *Zindagi Na Milegi Dobara*, *3 Idiots*, *Barfi!*

---

## 💬 Future Enhancements

* Add **audio-based emotion detection**
* Expand to **multi-face detection**
* Deploy on **cloud or mobile** platforms
* Integrate with streaming platforms

---

## 👤 Author

Built with ❤️ by Rucha Ahire — Passionate about AI, Computer Vision, and real-world problem solving.

