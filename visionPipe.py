import cv2 as cv
import tkinter as tk
from PIL import Image
from collections import deque
from transformers import pipeline
from emotionGraph import EmotionGraph

print("Hello Elif! Starting the emotion detector...")

# Loading the model
emotion_pipeline = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# Negative emotions to track
NEGATIVE_EMOTIONS = {"sad", "fear", "angry", "disgust"}

# Buffer for smoothing results
window_size = 30
emotion_window = deque(maxlen=window_size)

# Start webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# making emotion graph
root = tk.Tk()
graph = EmotionGraph(root)

# now continous using the webcam and taking the webcam frames !!!!
def update_frame():
    ret , frame  = cap.read()
    if not ret:
        print("Error in taking the frames from the webcam !!!! ")
        root.after(10, update_frame)

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    results = emotion_pipeline(pil_image)

    if results:
        top_prediction = results[0]
        label = top_prediction['label']
        score = top_prediction['score']

        emotion_window.append(label)

        if label == 'happy':
            color = (0, 255, 0)  # Green
        elif label == 'sad':
            color = (180, 130, 70)  # Steel Blue
        elif label == 'angry':
            color = (0, 0, 255)  # Red
        elif label == 'fear':
            color = (128, 0, 128)  # Purple
        elif label == 'neutral':
            color = (200, 200, 200)  # Grey
        elif label == 'disgust':
            color = (47, 107, 85)  # Olive
        else:
            color = (255, 255, 255)  # White


        cv.putText(frame, f"{label} ({score:.2f})", (30, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)


        negative_count = sum(1 for e in emotion_window if e in NEGATIVE_EMOTIONS)
        negative_ratio = negative_count / len(emotion_window)


        # Update graph
        graph.update(labels, score)

        if len(emotion_window) == window_size and negative_ratio > 0.6:
            cv.putText(frame, "!!! ALERT: Possible Distress !!!", (10, 220),
                       cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 5)


    cv.imshow("Emotion Detection", frame)

    # Quit on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        root.destroy()

    root.after(10, update_frame)

root.after(0, update_frame)
root.mainloop()

