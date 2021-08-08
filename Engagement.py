# import pyaudio
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import cv2
# import mediapipe as mp
# import os
# import json
# import numpy as np
# import csv
# from numpy import loadtxt
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import gdown
# import pickle


# np.set_printoptions(suppress=True) # don't use scientific notation
# CHUNK = 4096 # number of data points to read at a time
# RATE = 48000 # time resolution of the recording device (Hz)
# LAPSE = 60*10   # total number of seconds we want the camera to open
# WINDOW = 1   # number of seconds within a window
# FRAME_RATE = 4 # number of times we want to windowing the data in a second
# COORDINATES = ['x', 'y', 'z']


# def get_audio(stream, all_audio_keypoints):
#     data = np.fromstring(stream.read(CHUNK, exception_on_overflow=False),dtype=np.int16)
#     data = data * np.hanning(len(data)) # smooth the FFT by windowing data
#     fft = abs(np.fft.fft(data).real)
#     fft = fft[:int(len(fft)/2)] # keep only first half
#     freq = np.fft.fftfreq(CHUNK,1.0/RATE)
#     freq = freq[:int(len(freq)/2)] # keep only first half
#     freqPeak = freq[np.where(fft==np.max(fft))[0][0]]+1
#     amp=np.average(np.abs(data))*2
#     all_audio_keypoints['frequency'].append(freqPeak)
#     all_audio_keypoints['amplitude'].append(amp)
#     return


# def get_visual(cap, all_visual_keypoints):
#     success, image = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         # If loading a video, use 'break' instead of 'continue'.
#         return

#     # Flip the image horizontally for a later selfie-view display, and convert
#     # the BGR image to RGB.
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = holistic.process(image)

#     # Draw landmark annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
#     mp_drawing.draw_landmarks(
#         image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#     mp_drawing.draw_landmarks(
#         image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#     mp_drawing.draw_landmarks(
#         image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#     cv2.imshow('MediaPipe Holistic', image)

#     # Save the landmarks
#     if results.pose_landmarks:
#         j = 0
#         for data_point in results.pose_landmarks.landmark:
#             all_visual_keypoints['pose']['x'][j].append(data_point.x)
#             all_visual_keypoints['pose']['y'][j].append(data_point.y)
#             all_visual_keypoints['pose']['z'][j].append(data_point.z)
#             j += 1
#     else:
#         for j in range(33):
#             all_visual_keypoints['pose']['x'][j].append(np.nan)
#             all_visual_keypoints['pose']['y'][j].append(np.nan)
#             all_visual_keypoints['pose']['z'][j].append(np.nan)

#     if results.face_landmarks:   
#         j = 0
#         for data_point in results.face_landmarks.landmark:
#             all_visual_keypoints['face']['x'][j].append(data_point.x)
#             all_visual_keypoints['face']['y'][j].append(data_point.y)
#             all_visual_keypoints['face']['z'][j].append(data_point.z)
#             j += 1
#     else:
#         for j in range(468):
#             all_visual_keypoints['face']['x'][j].append(np.nan)
#             all_visual_keypoints['face']['y'][j].append(np.nan)
#             all_visual_keypoints['face']['z'][j].append(np.nan)


#     if results.left_hand_landmarks:
#         j = 0
#         for data_point in results.left_hand_landmarks.landmark:
#             all_visual_keypoints['left_hand']['x'][j].append(data_point.x)
#             all_visual_keypoints['left_hand']['y'][j].append(data_point.y)
#             all_visual_keypoints['left_hand']['z'][j].append(data_point.z)
#             j += 1
#     else:
#         for j in range(21):
#             all_visual_keypoints['left_hand']['x'][j].append(np.nan)
#             all_visual_keypoints['left_hand']['y'][j].append(np.nan)
#             all_visual_keypoints['left_hand']['z'][j].append(np.nan)

#     if results.right_hand_landmarks:
#         j = 0
#         for data_point in results.right_hand_landmarks.landmark:
#             all_visual_keypoints['right_hand']['x'][j].append(data_point.x)
#             all_visual_keypoints['right_hand']['y'][j].append(data_point.y)
#             all_visual_keypoints['right_hand']['z'][j].append(data_point.z)
#             j += 1
#     else:
#         for j in range(21):
#             all_visual_keypoints['right_hand']['x'][j].append(np.nan)
#             all_visual_keypoints['right_hand']['y'][j].append(np.nan)
#             all_visual_keypoints['right_hand']['z'][j].append(np.nan)

#     if cv2.waitKey(5) & 0xFF == 27:
#         return


# def process_visual(num, key, output_data, all_visual_keypoints, counter_head, counter_tail):
#     for coordinate in COORDINATES:
#         for i in range(num):
#             arr = np.array(all_visual_keypoints[key][coordinate][i][counter_head:counter_tail+1])
#             mean = np.nanmean(arr)
#             var = np.nanvar(arr)
#             if np.isnan(var):
#                 var = float("inf")
#             if np.isnan(mean):
#                 mean = float("inf")
#             output_data.append(mean)
#             output_data.append(var)


# def process_audio(key, counter_head, counter_tail, all_audio_keypoints):
#     arr = np.array(all_audio_keypoints[key][counter_head:counter_tail+1])
#     mean = np.nanmean(arr)
#     var = np.nanvar(arr)
#     return mean, var


# # calculate the mean and variance of the data within the last pre-set seconds
# def process_data(counter_head, counter_tail, all_visual_keypoints, all_audio_keypoints):
#     # process audio data
#     output_data = []
#     freq_mean, freq_var = process_audio('frequency', counter_head, counter_tail, all_audio_keypoints)
#     amp_mean, amp_var = process_audio('amplitude', counter_head, counter_tail, all_audio_keypoints)
#     # output_audio.append([freq_mean, freq_var, amp_mean, amp_var])
#     output_data.append(freq_mean)
#     output_data.append(freq_var)
#     output_data.append(amp_mean)
#     output_data.append(amp_var)

#     # process visual data
#     process_visual(33, 'pose', output_data, all_visual_keypoints, counter_head, counter_tail)
#     process_visual(468, 'face', output_data, all_visual_keypoints, counter_head, counter_tail)
#     process_visual(21, 'left_hand', output_data, all_visual_keypoints, counter_head, counter_tail)
#     process_visual(21, 'right_hand', output_data, all_visual_keypoints, counter_head, counter_tail)
    
#     return output_data

# def predict_engagement():
#     # create variables for holding input visual and audio data
#     all_visual_keypoints = {
#         'pose': {'x': [], 'y': [], 'z': []},
#         'face': {'x': [], 'y': [], 'z': []},
#         'left_hand': {'x': [], 'y': [], 'z': []},
#         'right_hand': {'x': [], 'y': [], 'z': []},
#     }
#     for coordinate in COORDINATES:
#         for i in range(33):
#             all_visual_keypoints['pose'][coordinate].append([])
#         for i in range(468):
#             all_visual_keypoints['face'][coordinate].append([])
#         for i in range(21):
#             all_visual_keypoints['left_hand'][coordinate].append([])
#             all_visual_keypoints['right_hand'][coordinate].append([])
#     all_audio_keypoints = {'frequency': [], 'amplitude': []}


#     # create variables for holding output visual and audio data
#     output_audio = []
#     output_visual = {
#         'pose': {'x': [], 'y': [], 'z': []},
#         'face': {'x': [], 'y': [], 'z': []},
#         'left_hand': {'x': [], 'y': [], 'z': []},
#         'right_hand': {'x': [], 'y': [], 'z': []},
#     }
#     for coordinate in COORDINATES:
#         for i in range(33):
#             output_visual['pose'][coordinate].append([])
#         for i in range(468):
#             output_visual['face'][coordinate].append([])
#         for i in range(21):
#             output_visual['left_hand'][coordinate].append([])
#             output_visual['right_hand'][coordinate].append([])

#     if not os.path.isfile('train_dataset.csv'):
#         print("downloading trainig data")
#         train_dataset_url = 'https://drive.google.com/u/0/uc?id=1MA4kLyYu_FWY2BLXvTnGxe7ffbnEmXmm&export=download'
#         download_filename = 'train_dataset.csv'
#         gdown.download(train_dataset_url, download_filename, quiet=False)
#         print("Finished downloading!")

#     # load data
#     dataset = loadtxt('train_dataset.csv', delimiter=",")

#     # hard-code y_train
#     y_train = np.zeros((3483,))
#     for i in range(len(y_train)):
#         if i < 496:
#             y_train[i] = 1
#         elif i < 2970 and i >= 1970:
#             y_train[i] = 1

#     if not os.path.isfile('model_pickle'):
#         x_train = dataset[:,0:3262]
#         model = XGBClassifier(use_label_encoder=False)
#         print("Modeling Training")
#         model.fit(x_train, y_train)
#         print("Training Done!")
#         with open('model_pickle', 'wb') as f:
#             pickle.dump(model, f)
#     else:
#         with open('model_pickle', 'rb') as f:
#             model = pickle.load(f)

#     # start the program
#     # mediapipe webcam setup
#     mp_drawing = mp.solutions.drawing_utils
#     mp_holistic = mp.solutions.holistic
#     holistic = mp_holistic.Holistic(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5)
#     cap = cv2.VideoCapture(0)

#     # PyAudio setup
#     p=pyaudio.PyAudio() 
#     stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
#                 frames_per_buffer=CHUNK) #uses default input device

#     counter_head = 0
#     counter_tail = 0
#     all_output = []
#     start = time.time()
#     end = start
#     interval = start

#     while cap.isOpened() and end-start <= LAPSE: #go for a few seconds
#         end = time.time()   
#         get_audio(stream, all_audio_keypoints)
#         get_visual(cap, all_visual_keypoints)
#         counter_tail += 1
#         if end-start > WINDOW:
#             counter_head += 1
        
#         if end-interval >= 1/FRAME_RATE:
#             interval = end
#             x_test = process_data(counter_head, counter_tail, all_visual_keypoints, all_audio_keypoints)
#             # all_output.append(x_test)
#             # make predictions for test data
#             x_test = np.array(x_test).reshape((1,-1))
#             y_pred = model.predict(x_test)
#             print("User Engagement", y_pred)
#             all_output.append(y_pred)
            
            
#     # close the stream gracefully
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     cap.release()

#     # debugging
#     # print('visual: ', len(all_visual_keypoints['pose']['x'][0]))
#     # print('audio: ', len(all_audio_keypoints['frequency']))
#     # np.savetxt("y_predict.csv", all_output, delimiter=",")
#     # print(len(all_output))

# if __name__=="__main__":
#     predict_engagement()


import mediapipe as mp
import pyaudio
import matplotlib.pyplot as plt
import time
import os
import json
import csv
import cv2
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gdown
import Data
import Audio
import Visual
import sys
from time import sleep
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import pickle
import subprocess
import pyttsx3

def predict_engagement():
    # engine = pyttsx3.init()
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', 'com.apple.speech.synthesis.voice.Victoria')

    np.set_printoptions(suppress=True) # don't use scientific notation
    CHUNK = 4096 # number of data points to read at a time
    RATE = 48000 # time resolution of the recording device (Hz)
    LAPSE = 60*10   # total number of seconds we want the camera to open
    WINDOW = 1   # number of seconds within a window
    FRAME_RATE = 4 # number of times we want to windowing the data in a second
    COORDINATES = ['x', 'y', 'z']

    # create variables for holding input visual and audio data
    all_visual_keypoints = {
        'pose': {'x': [], 'y': [], 'z': []},
        'face': {'x': [], 'y': [], 'z': []},
        'left_hand': {'x': [], 'y': [], 'z': []},
        'right_hand': {'x': [], 'y': [], 'z': []},
    }
    for coordinate in COORDINATES:
        for i in range(33):
            all_visual_keypoints['pose'][coordinate].append([])
        for i in range(468):
            all_visual_keypoints['face'][coordinate].append([])
        for i in range(21):
            all_visual_keypoints['left_hand'][coordinate].append([])
            all_visual_keypoints['right_hand'][coordinate].append([])
    all_audio_keypoints = {'frequency': [], 'amplitude': []}

    # create variables for holding output visual and audio data
    output_audio = []
    output_visual = {
        'pose': {'x': [], 'y': [], 'z': []},
        'face': {'x': [], 'y': [], 'z': []},
        'left_hand': {'x': [], 'y': [], 'z': []},
        'right_hand': {'x': [], 'y': [], 'z': []},
    }
    for coordinate in COORDINATES:
        for i in range(33):
            output_visual['pose'][coordinate].append([])
        for i in range(468):
            output_visual['face'][coordinate].append([])
        for i in range(21):
            output_visual['left_hand'][coordinate].append([])
            output_visual['right_hand'][coordinate].append([])

    if not os.path.isfile('train_dataset.csv'):
        print("downloading trainig data")
        train_dataset_url = 'https://drive.google.com/u/0/uc?id=1MA4kLyYu_FWY2BLXvTnGxe7ffbnEmXmm&export=download'
        download_filename = 'train_dataset.csv'
        gdown.download(train_dataset_url, download_filename, quiet=False)
        print("Finished downloading!")

    # load data
    dataset = loadtxt('train_dataset.csv', delimiter=",")

    # hard-code y_train
    y_train = np.zeros((3483,))
    for i in range(len(y_train)):
        if i < 496:
            y_train[i] = 1
        elif i < 2970 and i >= 1970:
            y_train[i] = 1

    x_train = dataset[:,0:3262]
    # fit model no training data

    if not os.path.isfile('model_pickle'):
        model = XGBClassifier(use_label_encoder=False)
        print("Modeling Training")
        model.fit(x_train, y_train)
        print("Training Done!")
        with open('model_pickle', 'wb') as f:
            pickle.dump(model, f)
    else:
        with open('model_pickle', 'rb') as f:
            model = pickle.load(f)

    # start the program
    # mediapipe webcam setup
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    # PyAudio setup
    p=pyaudio.PyAudio() 
    stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                frames_per_buffer=CHUNK) #uses default input device

    counter = 0
    counter_head = 0
    counter_tail = 0
    all_output = []
    start = time.time()
    end = start
    interval = start
    n=0

    while end-start <= LAPSE: #go for a few seconds
        end = time.time()
        Audio.get_audio(stream, all_audio_keypoints, CHUNK, RATE)
        Visual.get_visual(cap, all_visual_keypoints, holistic, mp_drawing, mp_holistic)
        counter_tail += 1
        if end-start > WINDOW:
            counter_head += 1
        
        if end-interval >= 1/FRAME_RATE:
            interval = end
            x_test = Data.process_data(counter_head, counter_tail, all_visual_keypoints, all_audio_keypoints, COORDINATES)
            # all_output.append(x_test)
            # make predictions for test data
            x_test = np.array(x_test).reshape((1,-1))
            y_pred = model.predict(x_test)
            all_output.append(y_pred)
            if counter==10:
                # if (y_pred==0):
                #     subprocess.call(['say', 'Focus up!'])
                print("User Engagement", y_pred)
                counter = 0
                return y_pred
            counter+=1
        n+=1

    # close the stream gracefully
    stream.stop_stream()
    stream.close()
    p.terminate()
    cap.release()

    # debugging
    # print('visual: ', len(all_visual_keypoints['pose']['x'][0]))
    # print('audio: ', len(all_audio_keypoints['frequency']))
    # np.savetxt("y_predict.csv", all_output, delimiter=",")
    # print(len(all_output))

if __name__=="__main__":
    predict_engagement()