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



app = Flask(__name__)

word = ' '
sentence = ' '

suggestions = []

def most_frequent(List):
    return max(set(List), key=List.count)

def gen_frames():  # generate frame by frame from camera
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
                    x_t = np.array(landmarks)

                    if x_t.flatten().shape[0] == 63:

                        res_ = model.predict(np.array([x_t]), verbose=0)

                        index_class = np.argmax(res_)

                        letter = alphabet[index_class]

                        letters.append(letter)
                    # Double hand special case
                    elif x_t.flatten().shape[0] == 126:

                        a = x_t[:21, :]

                        b = x_t[21:, :]

                        res_ = model.predict(np.array([a]), verbose=0)

                        index_class = np.argmax(res_)

                        output = alphabet[index_class]

                        res_1 = model.predict(np.array([b]), verbose=0)

                        index_class1 = np.argmax(res_1)

                        output1 = alphabet[index_class1]
                        # For fun
                        if output == 'Y' and output1 == 'Y':

                            letter = 'Sup Brah'

                        elif output == "V" and output1 == 'V':
                            letter = "Peace Among Worlds"

                # Flip the image horizontally for a selfie-view display.

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (100, 100)
                fontScale = 2
                fontColor = (0, 0, 0)
                thickness = 4
                lineType = 2

                image = cv2.putText(cv2.flip(image, 1), letter,
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)


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
alphabet = ['1', '2', '3', '4', 'space', 'del', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'
            , 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

autocomplete = autocomplete_factory(content_files=content_files)
model = tf.keras.models.load_model("src/app/model_saves/my_model/")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def most_frequent(List):
    return max(set(List), key=List.count)

def run_inference(image):
    global activate
    activate = 1
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        print(image.shape, image.dtype, type(image))
        # decode numpy array into OpenCV BGR image
        image = cv2.imdecode(image, flags=1)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        landmarks = []
        letter = ''
        if results.multi_hand_landmarks:
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
            x_t = np.array(landmarks)

            if x_t.flatten().shape[0] == 63 and activate:
                
                activate = 0
                res_ = model.predict(np.array([x_t]), verbose=0)

                index_class = np.argmax(res_)

                letter = alphabet[index_class]

                word.append(letter)

        else:
            activate = 1
    return letter

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recognize', methods=['POST'])
def recognize():
    global letters, word, suggestions, sentence

    sign = most_frequent(letters)

    occurrence = letters.count(sign)

    suggestions = autocomplete.search(word=word, size=3, max_cost=100)

    if occurrence < 10:
        sign = ''
    else:
        letters = ['']
        if sign == 'del':
            if len(sentence) <= 1:
                sentence = ' '
                word = ' '
            else:
                sentence = sentence[:-1]
                word = sentence.split(' ')[-1]
                word = word[:-1]
        elif sign == 'space':
            sentence += ' '
            word = ' '
            suggestions = []
        elif sign in ['1', '2', '3']:
            if len(suggestions) > int(sign)-1:
                sentence = sentence[:-len(word)+1]
                sentence += suggestions[int(sign)-1][0]
                sentence += ' ' 
                word = ' '
        else:
            word += sign
            sentence += sign

    print(sentence)
    
    # sign = 'hello world'  # return hello world for now
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
