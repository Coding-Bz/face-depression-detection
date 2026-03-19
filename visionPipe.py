import cv2 as cv
import tkinter as tk
from PIL import Image

from collections import deque
from transformers import pipeline
from emotionGraph import EmotionGraph

print("Hello Elif")


# loading the hugging face model for our test !!!
emotion_pipeline = pipeline("image-classification", model  = "dima806/facial_emotions_image_detection")


# making the tuple for checking the negative emotions 

NEGATIVE_EMOTIONS  = {"sad","fear","angry"}

# making the buffer for the capturing the frame 

window_size=  30 ;

emotion_window  = deque(maxlen=window_size)

# start the webcam for the taking the visual from the camera 

cap =  cv.VideoCapture(0)

# making emotion graph
root = tk.Tk()
graph = EmotionGraph(root)

# now continous using the webcam and taking the webcam frames !!!!
def update_frame():
    ret , frame  = cap.read()
    if not ret:
        print("Error in taking the frames from the webcam !!!! ")
        root.after(10, update_frame)


    # when i capture the video using the open cv it turn it 
    # BGR  and our model need i
    # RGB

    # converting the image into the n rgb for that !!! 
  
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)


    # Convert NumPy → PIL Image
    pil_image = Image.fromarray(rgb_frame)


    # now running the model 
    # results  =  emotion_pipeline(rgb_frame)
    # Run inference
    results = emotion_pipeline(pil_image)


    # now i will get the prediction for the facial data    !!!!     


    if results:
        labels =  results[0]['label']
        score = results[0]['score']

        emotion_window.append(labels)

        # Check negative emotion ratio
        negative_count = sum(1 for e in emotion_window if e in NEGATIVE_EMOTIONS)
        negative_ratio = negative_count / len(emotion_window)

        # Display current prediction
        cv.putText(frame, f"{labels} ({score:.2f})", (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        # Update graph
        graph.update(labels, score)

        # If consistently negative → show alert
        if negative_ratio > 0.6 and len(emotion_window) == window_size:
            cv.putText(frame, "⚠ ALERT: Possible Distress!", (20, 80),
                        cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)


    # Display the frame
    cv.imshow("Emotion Detection", frame)

    # Quit on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        root.destroy()

    root.after(10, update_frame)

root.after(0, update_frame)
root.mainloop()

