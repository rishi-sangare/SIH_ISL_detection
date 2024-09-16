# my_functions.py

import mediapipe as mp
import cv2
import numpy as np

def draw_landmarks(image, results):
    """
    Draw landmarks and connections on the image.

    Args:
        image (numpy.ndarray): The image on which landmarks will be drawn.
        results (mediapipe.python.solutions.holistic.Holistic): The results from MediaPipe Holistic.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

def image_process(image, model):
    """
    Process the image and obtain sign landmarks.

    Args:
        image (numpy.ndarray): The input image.
        model: The Mediapipe holistic object.

    Returns:
        results: The processed results containing sign landmarks.
    """
    # Set the image to read-only mode
    image.flags.writeable = False
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image using the model
    results = model.process(image)
    # Set the image back to writeable mode
    image.flags.writeable = True
    # Convert the image back from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def keypoint_extraction(results):
    """
    Extract the keypoints from the hand landmarks.

    Args:
        results: The processed results containing hand landmarks.

    Returns:
        keypoints (numpy.ndarray): The extracted keypoints for both hands.
    """
    # Initialize arrays for left and right hand keypoints with zeros
    lh = np.zeros(63)
    rh = np.zeros(63)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # MediaPipe does not provide handedness in landmarks. 
            # We assume the order of hands is left and then right if two hands are detected.
            # But this can be customized based on specific cases.
            handedness = results.multi_handedness[0].classification[0].label.lower() if results.multi_handedness else 'unknown'
            
            # Extract the keypoints and flatten them into a 1D array
            keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()

            # Assign the keypoints to the appropriate hand
            if handedness == 'left':
                lh = keypoints
            elif handedness == 'right':
                rh = keypoints

    # Concatenate the keypoints for both hands
    keypoints = np.concatenate([lh, rh])
    return keypoints

 