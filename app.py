from flask import Flask, render_template, request
import pickle
import numpy as np
import cv2
import tensorflow as tf

# Load the final model
model_file_path = 'model.pkl'

with open(model_file_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function for images
def preprocess_image(image):
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (400, 400))
    resized_image = resized_image.astype('float32') / 255.0  # Normalize pixel values
    return resized_image

# Function to make predictions
def predict_image(image):
    preprocessed_image = preprocess_image(image)
    # Reshape the image to match the model's expected input shape
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    # Predict class probabilities
    predicted_probabilities = loaded_model.predict(preprocessed_image)
    # Get the predicted class index
    predicted_class_index = np.argmax(predicted_probabilities)
    # Get the predicted class label
    classes_labels = ['hello', 'okay', 'peace', 'thumbs_up']
    predicted_label = classes_labels[predicted_class_index]
    # Get the certainty (probability) of the prediction
    certainty = predicted_probabilities[0][predicted_class_index] * 100
    return predicted_label, certainty

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['imagefile']
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)
        # Predict the class and certainty of the uploaded image
        predicted_label, certainty = predict_image(image)
        # Format the prediction result
        prediction_result = f"Hand Sign: {predicted_label} ({certainty:.2f}%)"
        return render_template('index.html', prediction=prediction_result)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
