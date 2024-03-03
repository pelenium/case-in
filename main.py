from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

model = YOLO("runs/detect/result/weights/best.pt")

while True:
    defectCounter = 0
    success, img = cap.read()
    img = cv2.flip(cv2.resize(img, (640, 640)), 1)
    results = model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            
            if cls == 0:
                defectCounter += 1

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0]*100))/100
                print("Defect found, Confidence --->", confidence)

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, "defect", org,
                            font, fontScale, color, thickness)

    cv2.putText(img, f"Defects in picture: {defectCounter}",
                [10, 20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
