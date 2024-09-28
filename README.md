# Sign-language-to-text-conversion

This project is a sign language to text conversion system that uses a pre-existing American Sign Language (ASL) dataset consisting of hand gestures for A-Z and 0-9. The system captures hand landmarks using MediaPipe and leverages machine learning to classify these gestures into their corresponding letters or numbers. It aims to facilitate communication for those using ASL by converting hand signs into text.


# Usage:

# 1. Training the Model
The model is trained using a pre-existing dataset of hand gesture images. Each image corresponds to a letter or number in ASL. The dataset is preprocessed to extract hand landmarks, which are used as input features for the classifier.

# 2. Running the Real-Time Detector
  To run the real-time sign language detector:
The system captures input from a webcam and processes the video stream frame by frame.
MediaPipe is used to detect hand landmarks in each frame.
The model predicts the corresponding letter or number based on the detected hand pose.
The output is displayed directly on the video feed, showing the predicted gesture and its associated letter or number.

# Steps to Run:
Clone the repository and navigate to the project directory.
Install the dependencies required.
Run the train_classifier.py to train the model using the provided ASL dataset.
Use inference_classifier.py to run real-time detection via webcam.

# Dependencies

OpenCV: For image capture and processing.

MediaPipe: For hand landmark detection.

NumPy: For data handling and manipulation.

Scikit-learn: For machine learning model training.

# Project Structure:

1.asl_dataset/: This is where the pre-existing ASL dataset is stored. Each sub-folder (e.g., A, B, C, …, Z) contains images representing hand gestures for each letter or number.

2.create_dataset.py: This script is responsible for creating the dataset used for training. It processes each image in the asl_dataset/ folder, extracts the hand landmarks using MediaPipe, and saves the processed data in data.pickle for model training.

3.data/: After processing the ASL images, the hand landmark data and their corresponding labels are saved here as data.pickle. This file is used during the training process.

4.train_classifier.py: This script handles training the model. It loads the processed data from data.pickle, trains a Random Forest classifier, evaluates the accuracy, and saves the trained model as model.p.

5.model.p: This is the trained machine learning model (Random Forest classifier) saved after running train_classifier.py. The model is used for real-time prediction in inference_classifier.py.

6.inference_classifier.py: This script handles real-time sign language detection. It uses a webcam to capture hand gestures, processes the frames to detect hand landmarks, uses the trained model (model.p) to predict the gesture, and displays the predicted letter or number on the screen.

