# Gesture Recognition Project

This project aims to recognize hand gestures from images using machine learning techniques. The system can classify hand gestures into four categories: "hello," "okay," "peace," and "thumbs up." The project involves the following components:

## Files:

1. **hand_sign.ipynb**: This Python script contains the code for data preprocessing, model training, evaluation, and prediction. It utilizes TensorFlow and OpenCV libraries for image processing, along with Matplotlib for visualization.

2. **app.py**: This Flask web application allows users to upload images and get predictions for the hand gestures detected in the images. It utilizes the trained model to make predictions.

3. **index.html**: This HTML file defines the structure of the web application's user interface. It includes input fields for uploading images and displays the uploaded image along with the prediction result.

## Dataset:

The dataset used for training and testing the model consists of hand gesture images categorized into four classes: "hello," "okay," "peace," and "thumbs up." The dataset contains a total of 600 images, with 150 images for each gesture class. Approximately 60-70% of the images are collected by the project creators, while the remaining 30-40% are sourced from external repositories such as Google Images.

The dataset is structured as follows:
- **data**: Main folder containing subfolders for each gesture class.
  - **hello**: Contains images depicting the "hello" hand gesture.
  - **okay**: Contains images depicting the "okay" hand gesture.
  - **peace**: Contains images depicting the "peace" hand gesture.
  - **thumbs_up**: Contains images depicting the "thumbs up" hand gesture.

## Model:

The model architecture used for gesture recognition is a Convolutional Neural Network (CNN) implemented using TensorFlow's Keras API. The CNN consists of convolutional layers followed by max-pooling layers to extract features from the input images. After several convolutional and pooling layers, the feature maps are flattened and fed into fully connected layers for classification. The output layer uses softmax activation to output probabilities for each gesture class.

## Usage:

To use the Gesture Recognition system:
1. Ensure all required libraries are installed (`tensorflow`, `opencv-python`, `matplotlib`, `flask`).
2. Run the `hand_sign.ipynb` in jupyter notebook to generate model.pkl file.
3. Run the `app.py` script to start the Flask web application.
4. Navigate to the provided URL in a web browser.
5. Upload an image containing a hand gesture.
6. Click the "Predict Image" button to obtain the prediction result.
7. The system will display the predicted hand gesture along with the certainty (probability) of the prediction.

## Contributors:

- <a href="https://github.com/Nikunj-Goswami4">**Nikunj Goswami**</a>
- <a href="https://github.com/Bansi5513">**Bansi Patel**</a>



For any questions, suggestions, or collaborations, please contact Nikunj Goswami or Bansi Patel.
