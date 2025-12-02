import cv2
from ultralytics import YOLO
import numpy as np
import json
import datetime
import os
from collections import deque, Counter

# --- COLAB PATH CONFIGURATION ---
INPUT_VIDEO_PATH = 'inVideo.mp4'
OUTPUT_VIDEO_PATH = 'output_yolov10_tracking.mp4'
OUTPUT_JSON_PATH = 'violations.json'

# --- MODEL CONFIG ---
# YOLOv10 supports native tracking.
MODEL_NAME = "yolov10l.pt"
CONFIDENCE_THRESHOLD = 0.5

# --- POLYGON CONFIG ---
REFERENCE_WIDTH = 2160
REFERENCE_HEIGHT = 3840

RAW_POLYGON_COORDS = [
   [991, 798],
   [1592, 783],
   [2151, 2105],
   [2131, 3816],
   [43, 3816]
]

# Class IDs
CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
TARGET_CLASS = 3
ALL_VEHICLES = [2, 3, 5, 7]
track_class_history = {}

# --- HELPER FUNCTIONS ---
def is_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def get_best_class(track_id, current_class=None):
    if track_id not in track_class_history:
        track_class_history[track_id] = deque(maxlen=30)
    if current_class is not None:
        track_class_history[track_id].append(current_class)
    if not track_class_history[track_id]:
        return None
    return Counter(track_class_history[track_id]).most_common(1)[0][0]

def scale_polygon(raw_coords, ref_w, ref_h, target_w, target_h):
    scale_x = target_w / ref_w
    scale_y = target_h / ref_h
    scaled_coords = []
    for x, y in raw_coords:
        new_x = int(x * scale_x)
        new_y = int(y * scale_y)
        scaled_coords.append([new_x, new_y])
    return np.array(scaled_coords, np.int32).reshape((-1, 1, 2))

# --- MAIN RUN ---
def run_detection():
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: File not found at {INPUT_VIDEO_PATH}")
        return

    print(f"Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate Polygon
    current_polygon = scale_polygon(RAW_POLYGON_COORDS, REFERENCE_WIDTH, REFERENCE_HEIGHT, width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print(f"Processing video: {width}x{height} using Native Tracking")

    violation_log = []
    logged_ids = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # --- NATIVE TRACKING ---
        # persist=True is REQUIRED for tracking to work between frames
        # tracker="bytetrack.yaml" is recommended for traffic (better than botsort)
        results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", conf=CONFIDENCE_THRESHOLD)

        # Get the boxes from the first result (since we only have 1 frame)
        boxes = results[0].boxes

        # Only proceed if there are detections with IDs
        if boxes.id is not None:
            # Extract attributes from GPU tensor to CPU list
            track_ids = boxes.id.int().cpu().tolist()
            cls_ids = boxes.cls.int().cpu().tolist()
            xyxys = boxes.xyxy.cpu().tolist()

            # Iterate through every tracked vehicle
            for box, track_id, cls in zip(xyxys, track_ids, cls_ids):

                # Filter by vehicle type
                if cls not in ALL_VEHICLES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1

                # --- LOGIC REMAINS THE SAME ---
                final_class_id = get_best_class(track_id, cls)
                if final_class_id is None: continue

                contact_point = (int((x1 + x2) / 2), int(y2))

                is_violation = False
                if final_class_id == TARGET_CLASS:
                    if is_inside_polygon(contact_point, current_polygon):
                        is_violation = True
                        if track_id not in logged_ids:
                            logged_ids.add(track_id)
                            timestamp = str(datetime.timedelta(seconds=frame_count/fps))
                            violation_log.append({
                                "timestamp": timestamp,
                                "vehicle_id": f"moto_{track_id}",
                                "type": "Lane Violation"
                            })
                            print(f"🚨 VIOLATION: Moto {track_id} at {timestamp}")

                # Drawing
                color = (0, 0, 255) if is_violation else (0, 255, 0)
                if final_class_id != TARGET_CLASS: color = (255, 0, 0)

                label = f"{CLASS_NAMES.get(final_class_id)} {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Overlay
        cv2.polylines(frame, [current_polygon], isClosed=True, color=(0, 0, 255), thickness=2)
        out.write(frame)

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(violation_log, f, indent=4)

    print(f"Done! Video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    run_detection()