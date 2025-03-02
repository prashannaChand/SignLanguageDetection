# Sign Language Recognition System

This project is a **Sign Language Recognition System** that recognizes alphabets using hand gestures. It utilizes **MediaPipe** for hand gesture detection and other machine learning techniques for recognizing and classifying the hand shapes corresponding to different letters of the alphabet.

## Introduction

This project aims to create a system capable of recognizing hand gestures for American Sign Language (ASL) alphabets. The system uses a combination of **MediaPipe's** hand gesture detection and custom machine learning models to detect and predict the hand shape corresponding to each letter of the alphabet. Currently, the system recognizes only the alphabets, and the goal is to expand it further for full words and phrases.

### Features:
- Recognizes hand gestures for each letter of the alphabet (A-Z).
- Uses the **MediaPipe** library for hand landmark detection and hand gesture recognition.
- Provides real-time predictions based on webcam input.

## Installation

### Prerequisites:
- Python 3.x
- pip (Python package installer)

### Steps to install:

1. Clone the repository:
    ```bash
    git clone https://github.com/prashannaChand/sign-language-recognition.git
    cd sign-language-recognition
    ```

2. Create a virtual environment (recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Make sure you have the following dependencies installed:
    - **MediaPipe** (for hand landmark detection)
    - **OpenCV** (for real-time video processing)
    - **TensorFlow** (for machine learning model)

    Example of installing required dependencies:
    ```bash
    pip install mediapipe opencv-python tensorflow
    ```

## Usage

1. Run the system:
    ```bash
    python sign_language_recognition.py
    ```

2. The system will open a window with real-time webcam input. Perform hand gestures in front of the camera to recognize the corresponding alphabet. The system will display the detected letter on the screen.

3. You can adjust the model or add more gestures for further alphabet recognition or implement a full sign language dictionary.

### Example Input:
- Raise your hand and make the shape for the letter "A" (closed fist with the thumb extended).
- The system should recognize and display the letter "A" on the screen.

### Outputs:
- The system will show the detected letter from the webcam input in real-time.

## Technologies Used

- **Python 3.x**: Programming language used for development.
- **MediaPipe**: Hand landmark and gesture recognition library.
- **OpenCV**: For video processing and image handling.
- **TensorFlow**: Used for any machine learning model training (if applicable).
- **NumPy**: For handling numerical data.

