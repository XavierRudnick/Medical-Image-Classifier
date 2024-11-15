import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
"""
This program trains a convolutional neural network using TensorFlow and Keras to classify 
lung disease images into four categories: COVID-19, Normal, Pneumonia, and Tuberculosis. 
Classifier uses EfficientNetB0 architecture with pre-trained ImageNet weights as the base model, 
which is fine-tuned on the custom lung disease dataset. The model is compiled with the Adam optimizer 
and trained with early stopping and learning rate reduction callbacks to prevent overfitting.
the model's performance is evaluated with a test dataset by a confusion matrix along with 
a classification report generated to visualize and assess the model's metrics.
"""

# ==============================
# Define constants and parameters
# ==============================

# List of class names corresponding to the target labels
class_names = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# Image dimensions - height, width, and channels in pixels
img_height = 150 
img_width = 150   
img_size = (150,150,3) 

# Batch size for training and validation datasets
batch_size = 32 

# Seed value for reproducibility of dataset shuffling and splitting  
seed = 42

# Directory containing the dataset of lung disease images
directory = 'lung_disease2'

# TensorFlow constant for optimizing data loading performance
AUTOTUNE = tf.data.AUTOTUNE

# =======================================
# Load and preprocess the image datasets
# =======================================

# Utilize Keras' image_dataset_from_directory to efficiently load images from the directory,
# automatically infer labels, and split the data into training and validation sets.

# Create the training dataset by loading images from the directory
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory,                          # Path to the dataset
    labels="inferred",                  # Automatically infer labels from subdirectory names
    label_mode="int",                   # Encode labels as integer indices
    batch_size=batch_size,              # Number of samples per batch
    image_size=(img_width,img_height),  # Resize images to a uniform size
    color_mode="rgb",                   # Load images in RGB format  
    validation_split=0.2,               # Reserve 20% of data for validation
    subset="training",                  # Specify this subset as training data
    shuffle=True,                       # Shuffle the data to ensure randomness
    seed=seed                           # Seed for reproducibility
)

# Create the validation (test) dataset similarly
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    batch_size=batch_size,
    image_size=(img_width,img_height),
    color_mode="rgb", 
    validation_split=0.2,
    subset="validation",
    shuffle=True,
    seed=seed
)

# ====================================================
# Prepare the datasets by normalizing the image pixel values
# ====================================================

# Define a function to preprocess images and labels
def prepare_dataset(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # Normalize pixel values from range [0, 255] to [0.0, 1.0]
    return image, label

# Apply the preprocessing function to the datasets
# Optimize performance by enabling parallel mapping, caching, and prefetching
train_data = train_data.map(prepare_dataset, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
test_data = test_data.map(prepare_dataset, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

# ===============================
# Define the convolutional neural network model
# ===============================

# Function to create the lung disease classification model
def lung_disease_model(input_shape, num_classes):

    # Load the EfficientNetB0 model with pre-trained ImageNet weights, excluding the top classification layers
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,          # Exclude top layers to add custom layers
        input_shape=input_shape,    # Input shape matching our data
        weights='imagenet'          # Use weights pre-trained on ImageNet dataset
    )

    # Freeze the base model to prevent weights from being updated during training
    base_model.trainable = False  

    # Define the input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Pass inputs through the base model
    x = base_model(inputs, training=False)

    # Apply global average pooling to reduce feature maps to a single vector per sample
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Add a fully connected layer with 128 units and ReLU activation
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    # Optionally, add a dropout layer for regularization if dropout_rate > 0
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.models.Model(inputs, outputs)
    return model

# Instantiate the model with the specified input size and number of classes
model = lung_disease_model(img_size, len(class_names))

# =======================================================
# Compile the model with optimizer, loss function, and metrics
# =======================================================

# Compile the model specifying the optimizer, loss function, and evaluation metrics
model.compile(
    optimizer='adam',                       # Adam optimizer for efficient gradient descent
    loss='sparse_categorical_crossentropy', # Loss function for multi-class classification with integer labels
    metrics=['accuracy']                    # Track accuracy during training and testing
)

# ======================
# Define training callbacks to prevent overfitting and improve training
# ======================


callbacks = [
    # EarlyStopping to halt training when validation loss stops improving
    tf.keras.callbacks.EarlyStopping(
        patience=3,                 # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True   # Restore the weights from the epoch with the best validation loss
    ),
    # ReduceLROnPlateau to reduce the learning rate when validation loss plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', # Monitor validation loss
        factor=0.2,         # Factor by which to reduce the learning rate
        patience=2,         # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-6,        # Lower bound on the learning rate
        verbose=1           # Verbosity mode
    )
]

# =======================
# Train the model on the training dataset and validate on the test dataset
# =======================

# Fit the model using the training dataset and validate using the test dataset
history = model.fit(
    train_data,                   # Training dataset
    validation_data=test_data,    # Validation dataset
    epochs=4,                   # Number of epochs to train (increase for better results)
    callbacks=callbacks         # Use the callbacks defined earlier
)

# =======================
# Evaluate the model's performance on the test dataset
# =======================

# Evaluate the trained model on the test dataset to get loss and accuracy
test_loss, test_acc = model.evaluate(test_data,verbose=1)
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)

# ===================================
# Make predictions and analyze the results
# ===================================

# Extract the true labels from the test dataset
# Concatenate labels from all batches into a single array
true_labels = np.concatenate(
    [y.numpy() for x, y in test_data], # Extract labels (y) and convert to NumPy arrays
    axis=0 # Concatenate along the first axis
)

# Use the predict function on test_data
predictions = model.predict(test_data)

# Convert probabilities to predicted class indices
predicted_classes = np.argmax(predictions, axis=1)

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_classes)

# Print classification report
print(classification_report(true_labels, predicted_classes, target_names=class_names))

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
