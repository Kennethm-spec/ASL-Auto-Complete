from fast_autocomplete import autocomplete_factory
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import cv2
import mediapipe as mp
import os
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)


content_files = {
    'words': {
        'filepath': 'src/app/word_dict.json',
        'compress': True  # means compress the graph data in memory
    }
}
alphabet = ['1', '2', '3', 'space', 'del', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
labels_ = ['1', '2', '3', 'space', 'del', 'A', 'B', 'C']
autocomplete = autocomplete_factory(content_files=content_files)
model = tf.keras.models.load_model("src/app/model/")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def most_frequent(List):
    return max(set(List), key=List.count)

def run_inference(image):
    print("here")
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
        print(image)
        if results.multi_hand_landmarks:
            print("Got hand")
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
    return letter

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    img_data = base64.b64decode(data['image'])
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    # print(f'here is img_array -> {img_array}')
    # print(f'{img_array.size}')
    # TODO: do something with image array
    if img_array.size == 0:
        sign = ''
        options = []
    else:
        sign = run_inference(img_array)
    options = autocomplete.search(word=sign, size=2, max_cost=100) # it returns an empty array if sign is empty
    # sign = 'hello world'  # return hello world for now
    return jsonify({'sign': sign, 'options': options})

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
