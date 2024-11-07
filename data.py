import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import time

#loading all image paths and labels in seperate arrays
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

#splitting dataframes into train and test groups
train_x, test_x, train_y, test_y = train_test_split(jpgs, label, test_size=0.1)

#convert dataframes to tensor flow obj
train_df = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_df = tf.data.Dataset.from_tensor_slices((test_x, test_y))

#classes of the dataframe
class_names = [
    "NORMAL",
    "PNEUMONIA",
]

#prepare dataset by converting to tensor obj, resizing, and normalizing
def prepare_dataset(path, label):
    image = tf.io.read_file(path) #load image
    image = tf.image.decode_jpeg(image, channels=1) #convert jpeg to tensor obj
    image = tf.image.resize(image, [150, 150]) #resize
    image = image / 255.0 #convert values to [0,1] range
    return image, label

#use tensorflow dynamic tuning to optimize the performance of pipeline
AUTOTUNE = tf.data.AUTOTUNE

#apply prepare_dataset function to each tensorflow obj, use num_parallel_calls to optimize preformence
train_df = train_df.map(prepare_dataset, num_parallel_calls=AUTOTUNE)
test_df = test_df.map(prepare_dataset, num_parallel_calls=AUTOTUNE)

BATCH_SIZE = 32

#shuffle to avoid unwanted trends in image order, group images in batches for better speeds, prefetch new batch
train_df = train_df.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_df = test_df.batch(BATCH_SIZE).prefetch(AUTOTUNE)


#creating model based on multiple previous medical image classifer medical papers with purpose of recognizing xrays
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
model.fit(train_df, epochs=5)

test_loss, test_acc = model.evaluate(test_df, verbose=1)
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)
