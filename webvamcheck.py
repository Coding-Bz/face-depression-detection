import cv2

# Open Iriun webcam (index 0 = /dev/video0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open Iriun webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Show the video stream
    cv2.imshow("Iriun Webcam", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
