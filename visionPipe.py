import cv2 as cv
import tkinter as tk
from PIL import Image
from collections import deque
from transformers import pipeline
from emotionGraph import EmotionGraph
from playsound import playsound
import os

print("Hello Elif! Starting the optimized emotion detector...")

# 1. Load everything ONCE (outside the loop)
emotion_pipeline = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

NEGATIVE_EMOTIONS = {"sad", "fear", "angry", "disgust"}
window_size = 20  # Smaller window responds faster
emotion_window = deque(maxlen=window_size)

cap = cv.VideoCapture(0)

root = tk.Tk()
graph = EmotionGraph(root)

def update_frame():
    ret, frame = cap.read()
    if not ret: root.after(10, update_frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        face_bgr = frame[y:y + h, x:x + w]

        face_rgb = cv.cvtColor(face_bgr, cv.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face_rgb)

        results = emotion_pipeline(pil_face)
        sad = "Sad.mp2"
        happy = "Happy.mp3"
        neutral = "Neutral.mp3"
        disgust = "Disgust.mp3"
        angry = "Angry.mp3"
        surprise = "Surprise.mp3"
        fear = "Fear.mp3"

        if results:
            label = results[0]['label']
            score = results[0]['score']
            emotion_window.append(label)

            color = (255, 255, 255)  # Default white
            if label == 'happy':
                color = (0, 255, 0)  # Green
                playsound(happy)
            elif label == 'sad':
                color = (180, 130, 70)  # Steel Blue
                playsound(sad)
            elif label == 'angry':
                color = (0, 0, 255)  # Red
                playsound(angry)
            elif label == 'fear':
                color = (128, 0, 128)  # Purple
                playsound(fear)
            elif label == 'neutral':
                playsound(neutral)
                color = (200, 200, 200)  # Grey
            elif label == 'disgust':
                playsound(disgust)
                color = (47, 107, 85)  # Olive
            elif label == 'surprise':
                playsound(surprise)
                color = (255, 165, 0) # Orange
            else:
                color = (255, 255, 255)  # White

            cv.putText(frame, f"{label}: {score:.2f}", (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Update graph
            graph.update(label, score)

    if len(emotion_window) == window_size:
        neg_ratio = sum(1 for e in emotion_window if e in NEGATIVE_EMOTIONS) / window_size
        if neg_ratio > 0.6:
            cv.putText(frame, "!!! DISTRESS ALERT !!!", (50, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)



    cv.imshow("Real-time Emotion Detection", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        root.destroy()

    root.after(10, update_frame)

root.after(0, update_frame)
root.mainloop()

