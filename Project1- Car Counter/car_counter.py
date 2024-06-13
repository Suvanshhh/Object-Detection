from ultralytics import YOLO
import cv2
import cvzone
import math

#FOR VIDEOS
cap = cv2.VideoCapture("../Test- Videos/cars.mp4")
#FOR VIDEOS

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

model = YOLO("yolov8n.pt")

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #BOUNDING BOX
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            #print(x1, y1, x2, y2)

            #THE ABOVE CODE HELPS PRINTING THE RECTANLGE BOUNDING BOX USING CV2.
            #THE BELOW CODE HELPS PRINTING THE STYLIZED RECTANLGE BOUNDING BOX USING CVZONE.

            w, h = x2-x1, y2-y1

            #CONFIDENCE
            conf = math.ceil((box.conf[0]*100))/100
            # cvzone.putTextRect(img, f'{conf}',(max(0, x1), max(35, y1)))

            #CLASS NAME
            #cls = box.cls[0]
            #cvzone.putTextRect(img, f'{cls}{conf}', (max(0, x1), max(35, y1)))
            # Till here it will just give us the ID of the class not the exact name.
            # Now for exact class name:
            cls = int(box.cls[0])
            currentClass = classNames[cls]

#CONDITIONING SO THAT ON CARS, MOTORBIKES, TRUCKS AND BUSES ARE DETECTED.
            if currentClass == "car" or currentClass== "truck" or currentClass=="bus" or currentClass=="motorbike" and conf > 0.3:
                cvzone.putTextRect(img, f'{currentClass}{conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2, offset=2)
                cvzone.cornerRect(img, (x1, y1, w, h), l=10)

            cv2.imshow("Image", img)
    cv2.waitKey(1)
