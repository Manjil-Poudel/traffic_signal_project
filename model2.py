import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- LOAD YOLO MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- VEHICLE CLASSES (COCO) ----------------
# car, motorbike, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# ---------------- TIME ALLOCATION FUNCTION ----------------
def allocate_time(vehicle_count):
    if vehicle_count <= 5:
        return 10
    elif vehicle_count <= 15:
        return 20
    elif vehicle_count <= 30:
        return 30
    else:
        return 40

# ---------------- LOAD THREE LANE IMAGES ----------------
image_paths = [
    "1st.png",
    "2nd.jpg",
    "3rd.png"
]

frames = []
vehicle_counts = []
allocated_times = []
results_list = []

for path in image_paths:
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {path}")

    frame = cv2.resize(frame, (640, 420))

    # YOLO detection
    results = model(frame, verbose=False)[0]

    count = 0
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in VEHICLE_CLASSES:
            count += 1

    frames.append(frame)
    vehicle_counts.append(count)
    allocated_times.append(allocate_time(count))
    results_list.append(results)

# ---------------- DECIDE GREEN SIGNAL ----------------
green_lane = np.argmax(vehicle_counts)

# ---------------- DRAW RESULTS ----------------
for i in range(3):
    frame = frames[i]
    results = results_list[i]

    # Draw bounding boxes
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Signal logic
    signal = "GREEN" if i == green_lane else "RED"
    signal_color = (0, 255, 0) if signal == "GREEN" else (0, 0, 255)

    # Overlay text
    cv2.putText(frame, f"Lane {i+1}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.putText(frame, f"Vehicles: {vehicle_counts[i]}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"Time: {allocated_times[i]} sec", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"Signal: {signal}", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, signal_color, 3)

    cv2.imshow(f"Lane {i+1}", frame)

print("------ TRAFFIC SIGNAL DECISION ------")
for i in range(3):
    print(f"Lane {i+1}: {vehicle_counts[i]} vehicles | {allocated_times[i]} sec")

print(f"\n🚦 GREEN SIGNAL → Lane {green_lane + 1}")

cv2.waitKey(0)
cv2.destroyAllWindows()
