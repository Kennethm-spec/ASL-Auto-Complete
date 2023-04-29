import cv2
import mediapipe as mp

# Define Mediapipe Hand detection module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load image
image = cv2.imread('hands.jpg')

# Convert image to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run hand detection
results = hands.process(image)

# Extract hand landmarks
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Calculate bounding box coordinates
        x_min = int(
            min([landmark.x for landmark in hand_landmarks.landmark]) * image.shape[1])
        x_max = int(
            max([landmark.x for landmark in hand_landmarks.landmark]) * image.shape[1])
        y_min = int(
            min([landmark.y for landmark in hand_landmarks.landmark]) * image.shape[0])
        y_max = int(
            max([landmark.y for landmark in hand_landmarks.landmark]) * image.shape[0])

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Display image
cv2.imshow('Hand detection', image)
cv2.waitKey(0)
