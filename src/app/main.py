import os
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

VIDEO_FOLDER = 'videos'
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
os.makedirs(VIDEO_FOLDER, exist_ok=True)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    video_data = base64.b64decode(data['video'])

    video_path = os.path.join(app.config['VIDEO_FOLDER'], 'asl_video.webm')

    with open(video_path, 'wb') as f:
        f.write(video_data)

    # Process the video with your ASL recognition model here
    result = process_video(video_path)  # Replace this function with the actual ASL recognition model
    return jsonify({'result': result})


def process_video(video_path):
    # Add your ASL recognition model code here
    result = "ASL recognition result"  # Replace this with the actual result from your model
    return result

# Disable caching for CSS files
@app.route('/static/css/<path:path>')
def send_css(path):
    return send_from_directory('static/css', path, cache_timeout=0)

if __name__ == '__main__':
    app.run(port=5003, debug=True)
