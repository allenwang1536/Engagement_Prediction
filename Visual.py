import numpy as np
import mediapipe as mp
import cv2

def get_visual(cap, all_visual_keypoints, holistic, mp_drawing, mp_holistic):
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        return

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(
    #     image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    # mp_drawing.draw_landmarks(
    #     image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(
    #     image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(
    #     image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # cv2.imshow('MediaPipe Holistic', image)

    # Save the landmarks
    if results.pose_landmarks:
        j = 0
        for data_point in results.pose_landmarks.landmark:
            all_visual_keypoints['pose']['x'][j].append(data_point.x)
            all_visual_keypoints['pose']['y'][j].append(data_point.y)
            all_visual_keypoints['pose']['z'][j].append(data_point.z)
            j += 1
    else:
        for j in range(33):
            all_visual_keypoints['pose']['x'][j].append(np.nan)
            all_visual_keypoints['pose']['y'][j].append(np.nan)
            all_visual_keypoints['pose']['z'][j].append(np.nan)

    if results.face_landmarks:   
        j = 0
        for data_point in results.face_landmarks.landmark:
            all_visual_keypoints['face']['x'][j].append(data_point.x)
            all_visual_keypoints['face']['y'][j].append(data_point.y)
            all_visual_keypoints['face']['z'][j].append(data_point.z)
            j += 1
    else:
        for j in range(468):
            all_visual_keypoints['face']['x'][j].append(np.nan)
            all_visual_keypoints['face']['y'][j].append(np.nan)
            all_visual_keypoints['face']['z'][j].append(np.nan)


    if results.left_hand_landmarks:
        j = 0
        for data_point in results.left_hand_landmarks.landmark:
            all_visual_keypoints['left_hand']['x'][j].append(data_point.x)
            all_visual_keypoints['left_hand']['y'][j].append(data_point.y)
            all_visual_keypoints['left_hand']['z'][j].append(data_point.z)
            j += 1
    else:
        for j in range(21):
            all_visual_keypoints['left_hand']['x'][j].append(np.nan)
            all_visual_keypoints['left_hand']['y'][j].append(np.nan)
            all_visual_keypoints['left_hand']['z'][j].append(np.nan)

    if results.right_hand_landmarks:
        j = 0
        for data_point in results.right_hand_landmarks.landmark:
            all_visual_keypoints['right_hand']['x'][j].append(data_point.x)
            all_visual_keypoints['right_hand']['y'][j].append(data_point.y)
            all_visual_keypoints['right_hand']['z'][j].append(data_point.z)
            j += 1
    else:
        for j in range(21):
            all_visual_keypoints['right_hand']['x'][j].append(np.nan)
            all_visual_keypoints['right_hand']['y'][j].append(np.nan)
            all_visual_keypoints['right_hand']['z'][j].append(np.nan)

    if cv2.waitKey(5) & 0xFF == 27:
        return

def process_visual(num, key, output_data, all_visual_keypoints, counter_head, counter_tail, COORDINATES):
    for coordinate in COORDINATES:
        for i in range(num):
            arr = np.array(all_visual_keypoints[key][coordinate][i][counter_head:counter_tail+1])
            mean = np.nanmean(arr)
            var = np.nanvar(arr)
            if np.isnan(var):
                var = float("inf")
            if np.isnan(mean):
                mean = float("inf")
            output_data.append(mean)
            output_data.append(var)