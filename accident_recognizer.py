import cv2
from ultralytics import YOLO
import math
import time

VIDEO_PATH = "accident.mp4"
ACCIDENT_IoU_THRESHOLD = 0.3
SPEED_DROP_THRESHOLD = 0.65   # 35% sudden slow-down
STOP_SPEED_THRESHOLD = 0.5
FRAMES_TO_CONFIRM = 10

model = YOLO("yolov8n.pt")  # small model, good enough

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(boxA_area + boxB_area - inter + 1e-6)

cap = cv2.VideoCapture(VIDEO_PATH)
mask = cv2.imread("mask_accident.png")
prev_positions = {}
prev_speeds = {}
accident_counter = {}

while True:
    ret, framee = cap.read()
    if not ret:
        break
    frame = cv2.bitwise_and(framee, mask)
    results = model.track(frame, persist=True, classes=[2,3,5,7], verbose=False)
    annotated = results[0].plot()

    vehicles = {}

    if results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue

            vid = int(box.id)
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            vehicles[vid] = [(x1,y1,x2,y2),(cx,cy)]

            cv2.putText(annotated, f"ID:{vid}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

    # SPEED + COLLISION LOGIC
    for id1 in vehicles:
        box1, (cx1,cy1) = vehicles[id1]

        if id1 in prev_positions:
            px,py = prev_positions[id1]
            speed = math.dist((cx1,cy1),(px,py))
        else:
            speed = 0

        prev_positions[id1] = (cx1,cy1)

        if id1 in prev_speeds:
            if prev_speeds[id1] != 0:
                drop_ratio = speed / (prev_speeds[id1]+1e-6)
            else:
                drop_ratio = 1
        else:
            drop_ratio = 1

        prev_speeds[id1] = speed

        # check collisions with other vehicles
        for id2 in vehicles:
            if id1 >= id2:
                continue
            box2,_ = vehicles[id2]

            overlap = iou(box1, box2)

            if overlap > ACCIDENT_IoU_THRESHOLD and drop_ratio < SPEED_DROP_THRESHOLD:
                accident_counter[(id1,id2)] = accident_counter.get((id1,id2),0)+1
                if accident_counter[(id1,id2)] > FRAMES_TO_CONFIRM:
                    cv2.putText(annotated,"ACCIDENT DETECTED!",
                                (50,50),cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,(0,0,255),4)
                    cv2.rectangle(annotated,(box1[0],box1[1]),
                                  (box1[2],box1[3]),(0,0,255),3)
                    cv2.rectangle(annotated,(box2[0],box2[1]),
                                  (box2[2],box2[3]),(0,0,255),3)

    cv2.imshow("Accident Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
