import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from fast_autocomplete import autocomplete_factory

content_files = {
    'words': {
        'filepath': 'word_dict.json',
        'compress': True  # means compress the graph data in memory
    }
}
alphabet = ['1', '2', '3', 'space', 'del', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
labels_ = ['1', '2', '3', 'space', 'del', 'A', 'B', 'C']
NUM_CLASSES = len(labels_)
model = Sequential()
# Create a checkpoint object and restore the weights
checkpoint_path = "model_checkpoints/cp_numbers_1.ckpt"

model.add(LSTM(64, return_sequences=True,
          activation='relu', input_shape=(21, 3)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

autocomplete = autocomplete_factory(content_files=content_files)
model.load_weights(checkpoint_path)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def most_frequent(List):
    return max(set(List), key=List.count)
