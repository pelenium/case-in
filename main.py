from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

model = YOLO("runs/detect/result/weights/best.pt")

while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 640))
    results = model(img, stream=True, verbose=False)
    # results = model.predict("test.jpg")

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            cls = "defect"
            print(cls)
            # 0 = human
    #         if cls == 0:
    #             peopleCounter += 1

    #             x1, y1, x2, y2 = box.xyxy[0]
    #             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    #             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    #             confidence = math.ceil((box.conf[0]*100))/100
    #             print("Person found, Confidence --->", confidence)

    #             org = [x1, y1]
    #             font = cv2.FONT_HERSHEY_SIMPLEX
    #             fontScale = 1
    #             color = (255, 0, 0)
    #             thickness = 2

    #             cv2.putText(img, "person", org,
    #                         font, fontScale, color, thickness)

    # cv2.putText(img, f"People in picture: {peopleCounter}",
    #             [10, 20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# from ultralytics import YOLO



# result = results[0]

# print(len(result.boxes))

# box = result.boxes[0]

# print("Object type:", box.cls)
# print("Coordinates:", box.xyxy)
# print("Probability:", box.conf)