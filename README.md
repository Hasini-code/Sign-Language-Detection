 # Sign Language Detection Using ML

## Overview
A system that detects hand signs from videos and predicts corresponding words to assist communication for deaf people.

## Features
- Trains a CNN model on hand gesture images
- Predicts hand signs from new videos
- Recognizes signs like YES, NO, HELLO, THANK YOU

## Technologies Used
Python, TensorFlow, Keras, OpenCV

## How to Run
1. Extract frames from your videos using `extract_frames.py`
2. Train the model using `train_model.py`
3. Test predictions using `predict_video.py`

## Note
- Data folder should contain frames extracted from videos
- For prediction, provide a new video file
