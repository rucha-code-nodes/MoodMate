import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Directory and classes
Datadirectory = "D:\\Journey To Speech Emotion Recognition\\1_Emotion Detection\\Data2"
Classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Define image size and batch size
img_size = 224
batch_size = 32

# Create ImageDataGenerator instance with enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Correct paths for training and validation subsets
train_generator = train_datagen.flow_from_directory(
    os.path.join(Datadirectory, 'train1'),
    target_size=(img_size, img_size),
    batch_size=batch_size,
    classes=Classes,
    class_mode='sparse',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join(Datadirectory, 'validation1'),
    target_size=(img_size, img_size),
    batch_size=batch_size,
    classes=Classes,
    class_mode='sparse',
    subset='validation'
)

# Load MobileNetV3 and define the new model
base_model = tf.keras.applications.MobileNetV3Small(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = True  # Unfreeze all layers for fine-tuning

base_input = base_model.input
base_output = base_model.output

final_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
final_output = tf.keras.layers.Dense(128, activation='relu')(final_output)
final_output = tf.keras.layers.Dense(64, activation='relu')(final_output)
final_output = tf.keras.layers.Dense(len(Classes), activation='softmax')(final_output)

new_model = tf.keras.models.Model(inputs=base_input, outputs=final_output)
new_model.summary()

# Compile the model with a learning rate scheduler
new_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

# Train the model
try:
    new_model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[lr_scheduler]
    )
except Exception as e:
    print(f"Error during model training: {e}")

# Save the model in Keras format
new_model.save('emotion_recognition_model.keras')

# Load the model for prediction
new_model = tf.keras.models.load_model('emotion_recognition_model.keras')

# Predicting on a new image
frame_path = "D:\\Journey To Speech Emotion Recognition\\1_Emotion Detection\\happyBoy.webp"
frame = cv2.imread(frame_path)

if frame is not None:
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        print("No faces detected")
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            facess = faceCascade.detectMultiScale(roi_gray)
            if len(facess) == 0:
                print("No faces detected in ROI")
            else:
                for (ex, ey, ew, eh) in facess:
                    face_roi = roi_color[ey:ey+eh, ex:ex+ew]

        if 'face_roi' in locals():
            final_image = cv2.resize(face_roi, (img_size, img_size))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image / 255.0

            Predictions = new_model.predict(final_image)
            print(Predictions[0])

            emotion_index = np.argmax(Predictions)
            emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
            print(emotions[emotion_index])
        else:
            print("Face ROI not found.")
else:
    print(f"Failed to load frame image: {frame_path}")



