import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

image_list = []

# Specify the folder path
folder_path = "chest_xray/train/NORMAL"
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpeg')]
n = len(image_files)
label = ["normal"] * n

folder_path2 = "chest_xray/train/PNEUMONIA"
image_files_pneumonia = [f for f in os.listdir(folder_path2) if f.endswith('.jpeg')]
m = len(image_files_pneumonia)
label_pneumonia = ["pneumonia"] * m

image_files.extend(image_files_pneumonia)
label.extend(label_pneumonia)

dict = {"filename" : image_files , "label": label}
df = pd.DataFrame(dict)


datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             validation_split=.2)



train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='chest_xray/combine/',
    x_col='filename',
    y_col='label',
    subset='training',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='binary',
    target_size=(150,150),
    validate_filenames=False
)
"""""
folder_path5 = "chest_xray/val/NORMAL"
image_files_val = [f for f in os.listdir(folder_path5) if f.endswith('.jpeg')]
a = len(image_files_val)
label_val = ["normal"] * a

folder_path6 = "chest_xray/val/PNEUMONIA"
image_files_val_pneumonia = [f for f in os.listdir(folder_path6) if f.endswith('.jpeg')]
b = len(image_files_val_pneumonia)
label_val_pneumonia = ["pneumonia"] * b

image_files_val.extend(image_files_val_pneumonia)
label_val.extend(label_val_pneumonia)

dict3 = {"filename" : image_files_val , "label": label_val}
df3 = pd.DataFrame(dict3)
"""""

valid_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='chest_xray/combine/',
    x_col='filename',
    y_col='label',
    subset='validation',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='binary',
    target_size=(150,150),
    validate_filenames=False
)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))# changed here
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator,epochs=5, validation_data=valid_generator,)



folder_path3 = "chest_xray/test/NORMAL"
image_files_test = [f for f in os.listdir(folder_path3) if f.endswith('.jpeg')]
j = len(image_files_test)
label_test = ["normal"] * j

folder_path4 = "chest_xray/test/PNEUMONIA"
image_files_test_pneumonia = [f for f in os.listdir(folder_path4) if f.endswith('.jpeg')]
k = len(image_files_test_pneumonia)
label_test_pneumonia = ["pneumonia"] * k


image_files_test.extend(image_files_test_pneumonia)
label_test.extend(label_test_pneumonia)

dict2 = {"filename" : image_files_test , "label": label_test}
df2 = pd.DataFrame(dict2)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df2,
    directory='chest_xray/combineTest',
    x_col='filename',
    y_col='label',
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode='binary',
    target_size=(150,150),
    validate_filenames=False
)

test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)