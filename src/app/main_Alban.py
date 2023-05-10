# Import libraries
from fast_autocomplete import autocomplete_factory
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import cv2
import mediapipe as mp
import os
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, Response

# Initialization of global variable
app = Flask(__name__)

word = ' '
sentence = ' '

suggestions = []

# Functions definition
def most_frequent(List):
    # Return the most frequent word of a list
    return max(set(List), key=List.count)

def gen_frames():  
    # We generate frame by frame from the camera
    # Inspired by https://github.com/NakulLakhotia/Live-Streaming-using-OpenCV-Flask/tree/main, https://github.com/nicknochnack/ActionDetectionforSignLanguage


    global letters

    letters = ['']
    while True:
        # Capture frame-by-frame
        cap = cv2.VideoCapture(0)

        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                landmarks = []
                letter = ''
                if results.multi_hand_landmarks:
                    # If we get a result, we draw the landmarks on the hand
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        # Get the bounding box of the hand
                        x_min, y_min, x_max, y_max = float('inf'), float(
                            'inf'), float('-inf'), float('-inf')
                        
                        for landmark in hand_landmarks.landmark:
                            x, y = landmark.x, landmark.y
                            x_min = min(x_min, x)
                            y_min = min(y_min, y)
                            x_max = max(x_max, x)
                            y_max = max(y_max, y)

                        # Normalize the coordinates with respect to the bounding box of the hand
                        for landmark in hand_landmarks.landmark:
                            x, y, z = landmark.x, landmark.y, landmark.z
                            x_norm = (x - x_min) / (x_max - x_min)
                            y_norm = (y - y_min) / (y_max - y_min)
                            landmarks.append([x_norm, y_norm, z])

                    # Convert the landmarks to a feature vector
                    feature_vector = np.array(landmarks)

                    # Get the size of the feature vector
                    feature_vector_shape = feature_vector.flatten().shape[0]

                    # Depending of the size of the feature vector, we can understand if we detected one hand (size = 63) or two hands (size = 126)
                    if feature_vector_shape == 63:
                        
                        # Predict
                        pred = model.predict(np.array([feature_vector]), verbose=0)

                        # Get the class of the prediction
                        index_class = np.argmax(pred)
                        
                        # Assign its corresponding label
                        letter = alphabet[index_class]
                        
                        # Append a letter to a batch of letters to take the most occurent later, which allows to increase the accuracy of the prediction
                        letters.append(letter)

                    # Double hand special case
                    elif feature_vector_shape == 126:
                        
                        # Get the feature of the two hands
                        hand_1 = feature_vector[:21, :]

                        hand_2 = feature_vector[21:, :]

                        # Make predictions
                        pred_1 = model.predict(np.array([hand_1]), verbose=0)

                        index_class_1 = np.argmax(pred_1)

                        letter_1 = alphabet[index_class_1]

                        pred_2 = model.predict(np.array([hand_2]), verbose=0)

                        index_class_2 = np.argmax(pred_2)

                        letter_2 = alphabet[index_class_2]

                        # Define special cases for fun
                        if letter_1 == 'Y' and letter_2 == 'Y':

                            letter = 'Sup Brah'

                        elif letter_1 == "V" and letter_2 == 'V':
                            letter = "Peace Among Worlds"

                # Write the label of the letter on the screen
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (100, 100)
                fontScale = 2
                fontColor = (255, 144, 30)
                thickness = 6
                lineType = 2
                
                # Put the text on the screen and flip the image to have a mirror effect
                image = cv2.putText(cv2.flip(image, 1), letter,
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)

                # Submit the video to the web application
                success, frame = cap.read()  # read the camera frame
                if not success:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', image)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

content_files = {
    'words': {
        'filepath': 'src/app/word_dict.json',
        'compress': True  # means compress the graph data in memory
    }
}

# Define alphabet that we are using
alphabet = ['1', '2', '3', '4', 'space', 'del', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'
            , 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Load weights of the pretrained model
autocomplete = autocomplete_factory(content_files=content_files)
model = tf.keras.models.load_model("src/app/model_saves/my_model/")

# Defining mediapipe variables
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define flask html pages
@app.route('/')
def index():
    return render_template('index.html')

# Define the recognize signs function
@app.route('/api/recognize', methods=['POST'])
def recognize():
    global letters, word, suggestions, sentence

    # We get data from the website, this activate variable allows to tell when we start recording
    data = request.get_json()

    activate = data['activate']

    if activate:
        letters = ['']

    # Get the most frequent sign amongst the registered  signs (allows better accuracy)
    sign = most_frequent(letters)

    # Occurrence of the symbols
    occurrence = letters.count(sign)

    # Calculate suggestions based on the letters typed
    suggestions = autocomplete.search(word=word, size=3, max_cost=100)

    # We check that we have a minimum number of occurences to print the symbol
    if occurrence < 15:

        # Delete charachter option
        if occurrence > 4 and sign == 'del':
            letters = ['']
            if len(sentence) <= 1:
                    sentence = ' '
                    word = ' '
            else:
                sentence = sentence[:-1]
                word = sentence.split(' ')[-1]
                word += ' '
        else:
            sign = ''


    else:
        letters = ['']

        # Different functionnalities depending of the symbol            
        if sign == 'space':
            sentence += ' '
            word = ' '
            suggestions = []
        elif sign == 'del':
            letters = ['']
            if len(sentence) <= 1:
                sentence = ' '
                word = ' '
            else:
                sentence = sentence[:-1]
                word = sentence.split(' ')[-1]
                word += ' '
        elif sign in ['1', '2', '3']:
            # Make a choice between the suggestions
            if len(suggestions) > int(sign)-1:
                sentence = sentence[:-len(word)+1]
                sentence += suggestions[int(sign)-1][0]
                sentence += ' ' 
                word = ' '
        else:
            word += sign
            sentence += sign
    
    # Return information to be displayed
    return jsonify({'sign': word.lower(), 'sentence': sentence.lower(), 'options': suggestions})

# Disable caching for CSS files
@app.route('/static/css/<path:path>')
def send_css(path):
    return send_from_directory('static/css', path, cache_timeout=0)

# Disable caching for JS files
@app.route('/static/js/<path:path>')
def send_js(path):
    return send_from_directory('static/js', path, cache_timeout=0)

if __name__ == '__main__':
    app.debug = True
    app.run(port=5003, debug=True)
