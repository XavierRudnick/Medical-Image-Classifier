import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import time
#putting all image paths and labels in seperate numpy arrays
folder_path = "chest_xray/train/NORMAL"
jpgs = [os.path.join(root, file)
        for root, dirs, files in os.walk(folder_path)
        for file in files
        if file.endswith('.jpeg')]
n = len(jpgs)
label = [0] * n

folder_path2 = "chest_xray/train/PNEUMONIA"
jpgs2 = [os.path.join(root, file)
        for root, dirs, files in os.walk(folder_path2)
        for file in files
        if file.endswith('.jpeg')]
m = len(jpgs2)
label2 = [1] * m
label.extend(label2)
jpgs.extend(jpgs2) 

#processing the images into arrays
array_list = [np.squeeze(tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(path, color_mode='grayscale',target_size=(150,150))), axis=-1) for path in jpgs]
arr = np.array(array_list)
label_np = np.array(label)

#splitting processed arrays into train and test groups
train_x, test_x, train_y, test_y = train_test_split(arr, label_np, test_size=0.1)


class_names = [
    "NORMAL",
    "PNEUMONIA",
]

train_x=train_x/255.0
test_x=test_x/255.0
#creating model
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation='softmax'),
])
#compiling with SparseCategoricalCrossentropy loss function
model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

model.summary()
model.fit(train_x, train_y, epochs=1)

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=1)
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)