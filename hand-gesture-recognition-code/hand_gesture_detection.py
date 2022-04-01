import mediapipe as mediapipe
import cv2 as cvision
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Loading the hand gesture recognize model
model = load_model('hand_gesture')

# Load class names
file = open('handGesture_names', 'r')
Gestures = file.read().split('\n')
file.close()
print(Gestures)

mediapipeHand = mediapipe.solutions.hands
hand = mediapipeHand.Hands(max_num_hands=1, min_detection_confidence=0.7)
mediaPipeDraw = mediapipe.solutions.drawing_utils


# Initialize the webcam
FrameCapture = cvision.VideoCapture(0)

def handProcess(VideoFrame):
     VideoFrame = cvision.flip(VideoFrame, 1)
     framergb = cvision.cvtColor(VideoFrame, cvision.COLOR_BGR2RGB)
     processed_result = hand.process(framergb)
     return processed_result

def GesturePredict(videoframe, x):
    mediaPipeDraw.draw_landmarks(videoframe, x, mediapipeHand.HAND_CONNECTIONS)
    # Predict gesture
    prediction = model.predict([positions])
    print(prediction)
    classID = np.argmax(prediction)
    gesture = Gestures[classID]
    print("Prediction=", prediction,"Resultant Gesture=", gesture)
    f = open("predicted.txt", "a")
    f.write(gesture)
    f.write("\n")
    f.close()
    return gesture


while True:
    # Read each VideoFrame from the webcam
    _, VideoFrame = FrameCapture.read()
    x, y, c = VideoFrame.shape
    processed_result = handProcess(VideoFrame)
    gesture = ''
    positions = []

    if processed_result.multi_hand_landmarks:
        positions = []
        for handslms in processed_result.multi_hand_landmarks:
            for mark in handslms.landmark:
                posX = int(mark.x * x)
                posY = int(mark.y * y)
                positions.append([posX, posY])
            gesture = GesturePredict(VideoFrame, handslms)

    # show the prediction on the VideoFrame
    cvision.putText(VideoFrame, gesture, (10, 50), cvision.FONT_HERSHEY_TRIPLEX,
                   1, (50, 52, 168), 2, cvision.LINE_AA)

    # Show the final output
    cvision.imshow("Output", VideoFrame)

    if ord('x') == cvision.waitKey(1):
        break