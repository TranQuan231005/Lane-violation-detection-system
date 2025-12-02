import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import json
import datetime
import os
from collections import deque, Counter

# --- CONFIGURATION ---
VIDEO_PATH = 'video.mp4'
OUTPUT_VIDEO = 'step2_output_fixed.mp4'
OUTPUT_JSON = 'violations.json'
CONFIDENCE_THRESHOLD = 0.4 

# --- QUAN TRỌNG: CẤU HÌNH TỌA ĐỘ GỐC ---
# Bạn cần điền kích thước của ảnh/video mà bạn đã dùng để lấy tọa độ này.
# Dựa vào tọa độ y=3816, mình đoán video gốc của bạn là 4K Dọc (2160x3840).
# Nếu sai, hãy sửa lại 2 số này cho đúng kích thước ảnh gốc bạn đã vẽ polygon.
REFERENCE_WIDTH = 2160  
REFERENCE_HEIGHT = 3840 

# Tọa độ gốc (sẽ được tự động co giãn)
RAW_POLYGON_COORDS = [
   [991, 798],
   [1592, 783],
   [2151, 2105],
   [2131, 3816],
   [43, 3816]
]

# Class IDs
CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
TARGET_CLASS = 3  # Motorcycle
ALL_VEHICLES = [2, 3, 5, 7]

# Tracking History
track_class_history = {} 

def is_inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0

def get_best_class(track_id, current_class=None):
    if track_id not in track_class_history:
        track_class_history[track_id] = deque(maxlen=30)
    if current_class is not None:
        track_class_history[track_id].append(current_class)
    if not track_class_history[track_id]:
        return None
    most_common = Counter(track_class_history[track_id]).most_common(1)[0][0]
    return most_common

# --- HÀM MỚI: TỰ ĐỘNG CO GIÃN POLYGON ---
def scale_polygon(raw_coords, ref_w, ref_h, target_w, target_h):
    """
    Co giãn tọa độ từ kích thước gốc sang kích thước video thực tế.
    """
    scale_x = target_w / ref_w
    scale_y = target_h / ref_h
    
    scaled_coords = []
    for x, y in raw_coords:
        new_x = int(x * scale_x)
        new_y = int(y * scale_y)
        scaled_coords.append([new_x, new_y])
        
    print(f"Scale Factor: X={scale_x:.2f}, Y={scale_y:.2f}")
    return np.array(scaled_coords, np.int32).reshape((-1, 1, 2))

def run_detection():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: {VIDEO_PATH} not found.")
        return

    # 1. Initialize Models
    print("Loading Models...")
    try:
        model = YOLO("yolov8l.pt") 
        print("Using YOLO Large")
    except:
        model = YOLO('yolov8n.pt')
        print("Using YOLO Nano")

    tracker = DeepSort(max_age=20, nms_max_overlap=1.0, nn_budget=100)

    # 2. Video Setup
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # --- BƯỚC QUAN TRỌNG: TÍNH TOÁN LẠI POLYGON ---
    print(f"Original Video Size: {width}x{height}")
    current_polygon = scale_polygon(RAW_POLYGON_COORDS, REFERENCE_WIDTH, REFERENCE_HEIGHT, width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    violation_log = []
    logged_ids = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # A. DETECT
        results = model(frame, stream=True, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf > CONFIDENCE_THRESHOLD and cls in ALL_VEHICLES:
                    detections.append([[x1, y1, w, h], conf, cls])

        # B. TRACK
        tracks = tracker.update_tracks(detections, frame=frame)

        # C. CHECK LOGIC
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            current_det_class = track.det_class if hasattr(track, 'det_class') else None
            final_class_id = get_best_class(track_id, current_det_class)

            if final_class_id is None:
                continue 

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            contact_point = (int((x1 + x2) / 2), int(y2))

            is_violation = False
            
            # Use 'current_polygon' instead of the global one
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

            # Draw
            label_text = CLASS_NAMES.get(final_class_id, "Unknown")
            if is_violation:
                color = (0, 0, 255) 
                label = f"VIOLATION {track_id}"
                cv2.circle(frame, contact_point, 8, color, -1)
            elif final_class_id == TARGET_CLASS:
                color = (0, 255, 255) 
                label = f"Moto {track_id}"
            else:
                color = (255, 0, 0) 
                label = f"{label_text} {track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # D. DRAW OVERLAY
        cv2.polylines(frame, [current_polygon], isClosed=True, color=(0, 0, 255), thickness=2)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [current_polygon], (0, 0, 50)) 
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        out.write(frame)
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(violation_log, f, indent=4)

    print("\n--- DONE ---")

if __name__ == "__main__":
    run_detection()