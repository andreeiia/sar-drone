from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model
model = YOLO("yolo26n.pt")

# Perform object detection on an image on class 0 (person)
# results = model("https://ultralytics.com/images/bus.jpg", classes = 0)

### Perform object detection on a video stream on class 0 (person)
## end by pressing q
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, img= cap.read()
    results = model(img, stream=True, classes = 0)

    for r in results:
        annotated_frame = r.plot() 
        cv2.imshow('Webcam', annotated_frame)

    x = cv2.waitKey(1)
    if x == ord('q') or x == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
## end video stream