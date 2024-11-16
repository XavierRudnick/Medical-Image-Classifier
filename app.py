from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os
"""
This Flask web application allows users to upload chest X-ray images 
to predict lung diseases using a pre-trained convolutional neural network model. 
The model classifies images into four categories: COVID-19, Normal, Pneumonia, and Tuberculosis. 
Uploaded images are preprocessed to match the model's input format, 
then predicted with confidence level being displayed to the user after.
"""

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained lung disease classification model
model = tf.keras.models.load_model('lung_classifier_model.keras')

# Define the class names corresponding to the model's output indices
class_names = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

# ====================================================
# Prepare the datasets by normalizing the image pixel values
# ====================================================

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)                     # Read the image file
    image = tf.image.decode_image(image, channels=3)        # Decode the image to a tensor with 3 color channels (RGB)
    image = tf.image.resize(image, [150, 150])              # Resize the image to the size expected by the model (150x150 pixels)
    image = tf.image.convert_image_dtype(image, tf.float32) # Normalize pixel values to the range [0, 1]
    image = tf.expand_dims(image, axis=0)                   # Add a batch dimension (required by the model)
    return image

# ====================================================
# Recieve uploaded image and test
# ====================================================

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # Check if a file was uploaded
        if 'file' not in request.files: 
            return 'No file part'
        
        file = request.files['file']
        
        # Ensure that a file has been selected
        if file.filename == '': 
            return 'Error : no file uploaded'

        if file: 
            # Save the uploaded file to a temporary directory
            filepath = os.path.join('uploads', file.filename) 
            file.save(filepath)

            # Preprocess the image to match model input requirements
            image = preprocess_image(filepath)

            # Use the model to predict the class of the lung disease
            predictions = model.predict(image)

            # Get and recieve the index of the class with the highest predicted probability
            predicted_class = class_names[np.argmax(predictions[0])]

           # Calculate the confidence percentage
            confidence = np.max(predictions[0]) * 100

            # Remove the uploaded file after processing
            os.remove(filepath)

            # Render the result page with the prediction and confidence
            return render_template('result.html', prediction=predicted_class, confidence=confidence)
        
    # For GET requests, render the upload form
    return render_template('index.html')

if __name__ == "__main__":
    # Create the 'uploads' directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    # Run the Flask application on port 5000 in debug mode
    app.run(port=5000)