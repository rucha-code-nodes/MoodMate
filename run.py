import tensorflow as tf
import cv2 
import os
import matplotlib.pyplot as plt
import numpy as np

img_array = cv2.imread("D:\\Journey To Speech Emotion Recognition\\1_Emotion Detection\\Data2\\train1\\angry\\Training_3908.jpg")
img_array.shape
plt.imshow(img_array)

Datadirectory = "D:\\Journey To Speech Emotion Recognition\\1_Emotion Detection\\Data2"

Classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]

for category in Classes:
  path = os.path.join(Datadirectory,category)
  for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,img))
    plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
    plt.show()
    break
  break


img_size = 224 # ImageNet 224x224
new_array = cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array,cv2.COLOR_BGR2RGB))
plt.show()

new_array.shape

# read all images and convert them all to array

training_Data = []  # data array
def create_training_Data():
    for category in Classes:
       path = os.path.join(Datadirectory,category)
       class_num = Classes.index(category) # 0 1, Label
       for img in os.listdir(path):
          try:
             img_array = cv2.imread(os.path.join(path,img))
             new_array = cv2.resize(img_array,(img_size,img_size))
             training_Data.append([new_array,class_num])

          except Exception as e:
             pass
create_training_Data()         
print(len(training_Data))

import random
random.shuffle(training_Data)

x = [] # data/feature
y = [] # Label

for features,label in training_Data:
   x.append(features)
   y.append(label)

x = np.array(x).reshape(-1,img_size,img_size,3) # converting it to 4 dimension
x.shape


# normalizing data 
x = x/255.0

y[0]

Y = np.array(y)
Y.shape

# DEEP LEARNING MODEL FOR TRANING - TRANSFER LEARNING 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.applications.MobileNetV3()  # pretaitrained model
model.summary()

# TRANSFER LEARNING - TUNING ,WEIGHTS WILL STRAT FROM LAST CHECK POINT

base_input = model.layers[0].input
base_output = model.layers[0].output

final_output = layers.Dense(128)(base_output) # ADDING NEW LAYER AFTER THE OUTPUT OF GLOBAL POOLING LAYER
final_output = layers.Activation('relu')(final_output) #ACTIVATION FUNC
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7,activation='softmax')(final_output) # 7 AS MY CLASSES ARE 7 AND CLASSIFICATION LAYERS

final_output


new_model = keras.Model(inputs = base_input, outputs = final_output)
new_model.summary()

new_model.compile(loss = "sparse_categorical_crossentropy",optimizer = "adam", metrics = ["accuracy"])

new_model.fit(x,Y,epochs = 25)
new_model.save('my_model_64p35.h5')
new_model = tf.keras.models.load_model('Final_model_95p07.h5')







