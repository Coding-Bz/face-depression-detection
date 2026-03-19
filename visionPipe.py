import cv2 as cv
import tkinter as tk
from PIL import Image
from collections import deque
from transformers import pipeline
from emotionGraph import EmotionGraph
import threading
import os
import time
import pygame

print("Hello Elif! Optimierung läuft...")

# 1. Initialisierung
pygame.mixer.init()
emotion_pipeline = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

NEGATIVE_EMOTIONS = {"sad", "fear", "angry", "disgust"}
window_size = 20
emotion_window = deque(maxlen=window_size)

SOUND_FILES = {
    "happy": "Happy.mp3",
    "sad": "Sad.mp2",
    "neutral": "Neutral.mp3",
    "disgust": "Disgust.mp3",
    "angry": "Angry.mp3",
    "surprise": "Surprise.mp3",
    "fear": "Fear.mp3"
}

# Variablen für die Stabilisierung
last_played_emotion = None
current_label = "Scanning..."
current_score = 0.0
current_color = (255, 255, 255)
current_faces = []

frame_count = 0
cap = cv.VideoCapture(0)
root = tk.Tk()
graph = EmotionGraph(root)


def play_sound_async(label):
    global last_played_emotion

    if label == last_played_emotion:
        return

    if label in SOUND_FILES and os.path.exists(SOUND_FILES[label]):
        try:
            pygame.mixer.music.load(SOUND_FILES[label])
            pygame.mixer.music.play()
            last_played_emotion = label  # Merken, was zuletzt gespielt wurde
        except Exception as e:
            print(f"Sound Error: {e}")


def update_frame():
    global frame_count, current_label, current_score, current_color, current_faces

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame_count += 1

    if frame_count % 5 == 0:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        current_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in current_faces:
            face_rgb = cv.cvtColor(frame[y:y + h, x:x + w], cv.COLOR_BGR2RGB)
            pil_face = Image.fromarray(face_rgb)
            results = emotion_pipeline(pil_face)

            if results:
                current_label = results[0]['label']
                current_score = results[0]['score']
                emotion_window.append(current_label)

                # Farben setzen
                if current_label == 'happy':
                    current_color = (0, 255, 0)
                elif current_label == 'sad':
                    current_color = (180, 130, 70)
                elif current_label == 'angry':
                    current_color = (0, 0, 255)
                elif current_label == 'fear':
                    current_color = (128, 0, 128)
                elif current_label == 'neutral':
                    current_color = (200, 200, 200)
                elif current_label == 'disgust':
                    current_color = (47, 107, 85)
                elif current_label == 'surprise':
                    current_color = (255, 165, 0)

                # Sound nur bei Änderung abspielen
                threading.Thread(target=play_sound_async, args=(current_label,), daemon=True).start()
                graph.update(current_label, current_score)


    for (x, y, w, h) in current_faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(frame, f"{current_label}: {current_score:.2f}", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, current_color, 2)


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
        return

    root.after(10, update_frame)


root.after(0, update_frame)
root.mainloop()