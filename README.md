# Engagement_Prediction

Modularized engagement detection machine learning model (XGBoost Classifier taking in audio/visual input) 

Audio data is captured in chunks using PyAudio. Then, each chunk undergoes a Fast Fourier Transform to convert time-domain signal to frequency domain. The dominant frequency and amplitude of each audio chunk are calculated and stored.
Video frames are captured using OpenCV. Each frame is processed using MediaPipe Holistic to extract keypoints for face, hands, and pose. These keypoints (coordinates x, y, z) are stored in all_visual_keypoints.
The program then calculates the mean and variance of both audio and visual keypoints within a specified window. The output is a feature vector representing the mean and variance of keypoints and audio characteristics over the specified time window.
As the program runs, it continually captures audio and visual data, extracts features, and uses the trained model to predict user engagement in real-time.
