import os
import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
from itertools import product
import keyboard
from my_functions import keypoint_extraction  # Assuming you have this function in 'my_functions.py'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
import language_tool_python



# Define Streamlit layout with tabs
st.title("Hand Sign Recognition Application")
tab1, tab2 , tab3= st.tabs(["Data Collection", "Model Training","Sign Language Detector"])

# Data Collection tab
with tab1:
    st.header("Hand Sign Data Collection")
    st.write("Enter the actions (comma-separated) that you want to record hand landmarks for:")

    # Input for actions
    action_input = st.text_input("Actions (comma-separated)", "Stay,Happy")
    actions = np.array([action.strip() for action in action_input.split(',')])

    # Input for number of sequences and frames per action
    sequences = st.number_input("Number of Sequences", min_value=1, max_value=100, value=30)
    frames = st.number_input("Frames per Sequence", min_value=1, max_value=100, value=10)

    # Set the path where the dataset will be stored
    PATH = os.path.join('data')   

    if st.button("Start Data Collection"):
        for action, sequence in product(actions, range(sequences)):
            try:
                os.makedirs(os.path.join(PATH, action, str(sequence)))
            except:
                pass

        # Initialize MediaPipe Hand Landmarker
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        # Set up hand detection and tracking
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            cap = cv2.VideoCapture(0)  # Start video capture
            stframe = st.empty()  # For displaying the camera stream in Streamlit

            for action, sequence, frame in product(actions, range(sequences), range(frames)):
                if frame == 0:
                    while True:
                        success, image = cap.read()
                        if  not success:
                            st.error("Failed to capture frame")
                            continue

                        image = cv2.flip(image, 1)  # Flip the image horizontally
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image_rgb.flags.writeable = False

                        # Process the image and detect hands
                        results = hands.process(image_rgb)

                        image_rgb.flags.writeable = True
                        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Display instructions on the image
                        cv2.putText(image_bgr, f"Recording data for '{action}'. Sequence {sequence}.",
                                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(image_bgr, 'Press "Space" when ready.', 
                                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                        # Update the Streamlit image
                        stframe.image(image_bgr, channels="BGR")

                        if keyboard.is_pressed(' '):
                            break

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            exit()
                else:
                    success, image = cap.read()
                    if not success:
                        st.error("Failed to capture frame")
                        continue

                    image = cv2.flip(image, 1)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = hands.process(image_rgb)

                    image_rgb.flags.writeable = True
                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    stframe.image(image_bgr, channels="BGR")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Extract landmarks and save to arrays
                keypoints = keypoint_extraction(results)
                frame_path = os.path.join(PATH, action, str(sequence), str(frame))
                np.save(frame_path, keypoints)

            cap.release()
            cv2.destroyAllWindows()

        st.success("Data collection completed!")

# Model Training tab
with tab2:
    st.header("Train the Model")



    if st.button("Start Training"):
        # Set the path to the data directory
        PATH = os.path.join('data')
        # Check if the data directory exists
        if not os.path.exists(PATH):
            st.error("Data directory not found. Please collect data in Tab 1 first.")
        else:
            try:
                # Create an array of actions (signs)
                actions = np.array(os.listdir(PATH))

                if len(actions) == 0:
                    st.error("No action directories found in the data folder. Please collect data in Tab 1 first.")
                else:
                    # Define sequences and frames
                    sequences = 30
                    frames = 10

                    # Label map to map actions to numeric values
                    label_map = {label:num for num, label in enumerate(actions)}

                    # Load landmarks and corresponding labels
                    landmarks, labels = [], []
                    for action, sequence in product(actions, range(sequences)):
                        temp = []
                        for frame in range(frames):
                            npy_path = os.path.join(PATH, action, str(sequence), str(frame) + '.npy')
                            if not os.path.exists(npy_path):
                                raise FileNotFoundError(f"File not found: {npy_path}")
                            npy = np.load(npy_path)
                            temp.append(npy)
                        landmarks.append(temp)
                        labels.append(label_map[action])

                    X, Y = np.array(landmarks), to_categorical(labels).astype(int)

                    # Split the data into training and testing sets
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

                    # Define the model architecture
                    model = Sequential()
                    model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(10, 126)))
                    model.add(LSTM(64, return_sequences=True, activation='relu'))
                    model.add(LSTM(32, return_sequences=False, activation='relu'))
                    model.add(Dense(32, activation='relu'))
                    model.add(Dense(actions.shape[0], activation='softmax'))

                    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

                    # Placeholder for logs
                    log_placeholder = st.empty()

                    # Function to log epochs
                    class StreamlitLogger(Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            log_placeholder.text(f"Epoch {epoch + 1}: {logs}")

                    # Train the model and display logs in Streamlit
                    model.fit(X_train, Y_train, epochs=100, callbacks=[StreamlitLogger()])

                    model.save('my_model.h5')

                    # Make predictions and calculate accuracy
                    predictions = np.argmax(model.predict(X_test), axis=1)
                    test_labels = np.argmax(Y_test, axis=1)
                    accuracy = metrics.accuracy_score(test_labels, predictions)

                    st.success(f"Model trained with accuracy: {accuracy:.2f}")

            except FileNotFoundError as e:
                st.error(f"Error: {str(e)}. Please ensure you have collected data for all actions in Tab 1.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
       
with tab3:
    st.header("Sign Recognition")
    
    if st.button("Start Recognization"):

        # Initialize the necessary components
        PATH = os.path.join('data')
        actions = np.array(os.listdir(PATH))
        model = load_model('my_model.h5')
        tool = language_tool_python.LanguageToolPublicAPI('en-UK')

        # Placeholder for displaying the camera stream
        stframe = st.empty()

        # Access the camera
        cap = cv2.VideoCapture(0)
        sentence, keypoints, last_prediction, grammar_result = [], [], None, ""

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break

                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                results = hands.process(image_rgb)

                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                keypoints.append(keypoint_extraction(results))

                if len(keypoints) == 10:
                    keypoints_array = np.array(keypoints)
                    prediction = model.predict(np.expand_dims(keypoints_array, axis=0))
                    predicted_action = actions[np.argmax(prediction)]

                    if np.max(prediction) > 0.9 and predicted_action != last_prediction:
                        sentence.append(predicted_action)
                        last_prediction = predicted_action

                    keypoints = []

                if len(sentence) > 7:
                    sentence = sentence[-7:]

                if keyboard.is_pressed(' '):
                    sentence, keypoints, last_prediction, grammar_result = [], [], None, ""

                if sentence:
                    sentence[0] = sentence[0].capitalize()

                if len(sentence) >= 2 and sentence[-1].isalpha() and sentence[-2].isalpha():
                    sentence[-2] += sentence[-1]
                    sentence.pop()

                if keyboard.is_pressed('enter'):
                    text = ' '.join(sentence)
                    grammar_result = tool.correct(text)

                display_text = grammar_result if grammar_result else ' '.join(sentence)
                textsize = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_X_coord = (image_bgr.shape[1] - textsize[0]) // 2
                cv2.putText(image_bgr, display_text, (text_X_coord, 470), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                stframe.image(image_bgr, channels="BGR")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        tool.close()
