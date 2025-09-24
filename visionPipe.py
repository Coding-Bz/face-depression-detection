import cv2 as cv 
from PIL import Image

from collections import deque
from transformers import pipeline

print("Hello aman deep")


# loading the hugging face model for our test !!!
emotion_pipeline = pipeline("image-classification", model  = "dima806/facial_emotions_image_detection")


# making the tuple for checking the negative emotions 

NEGATIVE_EMOTIONS  = {"sad","fear","angry"}

# making the buffer for the capturing the frame 

window_size=  30 ; 

emotion_window  = deque(maxlen=window_size)

# start the webcam for the taking the visual from the camera 

cap =  cv.VideoCapture(0)


# now continous using the webcam and taking the webcam frames !!!!
while(1):
    ret , frame  = cap.read()
    if not ret:
        print("Error in taking the frames from the webcam !!!! ")
        break ; 


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
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # If consistently negative → show alert
        if negative_ratio > 0.6 and len(emotion_window) == window_size:
            cv.putText(frame, "⚠ ALERT: Possible Distress!", (20, 80),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


    # Display the frame
    cv.imshow("Emotion Detection", frame)

    # Quit on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
