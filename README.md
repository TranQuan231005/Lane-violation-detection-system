# Lane-violation-detection-system
------------------------------------------------------------------------

# 1. Introduction

This project implements a real-time Lane Violation Detection System
using computer vision and deep learning. The system detects vehicles,
tracks their movements, models lane boundaries, and identifies when
motorcycles illegally enter restricted lanes. It outputs a processed
video with bounding boxes and a JSON file containing violation logs
(timestamp, vehicle ID).

------------------------------------------------------------------------

# 2. Objectives

-   Detect vehicles in video streams using YOLOv10.
-   Track each vehicle consistently across frames.
-   Estimate lane boundaries using polygon-based region annotation.
-   Identify lane violations for motorcycles.
-   Record detected violations with timestamps.
-   Generate an annotated output video.

------------------------------------------------------------------------

# 3. System Architecture

              ┌──────────────────┐
              │   Input Video    │
              └────────┬─────────┘
                       │
                       ▼
            ┌───────────────────────┐
            │     Preprocessing     │
            │ - Load video          │
            │ - Scale polygon       │
            └─────────┬─────────────┘
                      │
                      ▼
            ┌───────────────────────┐
            │ Object Detection       │
            │ (YOLOv10)              │
            └─────────┬─────────────┘
                      │
                      ▼
            ┌───────────────────────┐
            │ Multi-object Tracking │
            │   (ByteTrack)         │
            └─────────┬─────────────┘
                      │
                      ▼
            ┌───────────────────────┐
            │ Lane Violation Logic  │
            │ - Contact point check │
            │ - Polygon test        │
            └─────────┬─────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │  Outputs                     │
       │ - Annotated video            │
       │ - JSON violation logs        │
       └──────────────────────────────┘

------------------------------------------------------------------------

# 4. Technologies Used

  Component       Description
  --------------- -----------------------------------------------------
  **YOLOv10**     Object detection model with native tracking support
  **ByteTrack**   Multi-object tracker for stable ID assignment
  **OpenCV**      Video reading, writing, drawing, polygon operations
  **NumPy**       Polygon scaling and geometry operations
  **Python 3**    Main programming environment
  **JSON**        Logging violation events

Additional parameters: - Confidence threshold: **0.5** - Target
violation class: **Motorcycle (ID = 3)**

------------------------------------------------------------------------
## 4.1 Setup Environment (Google Colab)
Go to https://colab.research.google.com/
Select New Notebook

### 4.1.1 Install Required Libraries

Paste this code into the block
!pip install ultralytics
!pip install opencv-python
!pip install matplotlib

### 4.1.2 Display Input, Output Video inside Colab

<img width="596" height="666" alt="image" src="https://github.com/user-attachments/assets/fd228e04-0da3-4f95-bb74-22855a611f0d" />

VIDEO_PATH = '/content/video.mp4'
OUTPUT_VIDEO = '/content/step2_output.mp4'
OUTPUT_JSON = '/content/violations.json'

# 5. Dataset

## 5.1 Dataset Type

The system works with **video-based input**, not training datasets.\
It operates on: - Traffic surveillance footage - Videos from
fixed/moving cameras - Multi-lane roads with mixed vehicle types

## 5.2 Input Video Characteristics

  Property     Description
  ------------ ----------------------------------------------
  Format       `.mp4`
  Resolution   Read dynamically, polygon scaled accordingly
  Frame Rate   Extracted using OpenCV
  Content      Vehicles on road lanes
  Duration     Any length supported

The input is defined as:

    INPUT_VIDEO_PATH = 'inVideo.mp4'

## 5.3 Polygon Annotation Dataset

The polygon represents the restricted lane area:

    RAW_POLYGON_COORDS = [
        [991, 798],
        [1592, 783],
        [2151, 2105],
        [2131, 3816],
        [43, 3816]
    ]

-   Annotation type: List of (x, y) coordinates\
-   Reference resolution: **2160 × 3840**\
-   Auto-scaled to video resolution

Used to determine motorcycle lane violations.

## 5.4 Vehicle Class Labels

  Class ID   Label
  ---------- ------------
  2          Car
  3          Motorcycle
  5          Bus
  7          Truck

Target violation class:

    TARGET_CLASS = 3

## 5.5 Ground Truth Assumptions

-   YOLOv10 predictions serve as "ground truth" for vehicle detection.
-   ByteTrack ensures identity consistency.
-   Polygon serves as lane boundary ground truth.
-   Violation definition:\
    **A motorcycle is violating if its bottom-center point enters the
    polygon area.**

## 5.6 Suggested Dataset Structure (Optional)

    dataset/
    │── videos/
    │     ├── inVideo.mp4
    │     ├── intersection_1.mp4
    │     └── highway_lane_3.mp4
    │
    │── annotations/
    │     ├── polygon_1.json
    │     ├── polygon_2.json
    │     └── readme.md
    │
    └── evaluation/
          ├── gt_violations.json
          └── metrics.csv


# 6. Implementation Details

## 6.1 Video Processing

-   Frames read using OpenCV.
-   Output video written in `.mp4v` codec.
-   FPS and resolution preserved.

## 6.2 Object Detection & Tracking

Using YOLOv10 native tracking:

    results = model.track(
        frame, 
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE_THRESHOLD
    )

Each tracked object includes: - Bounding box\
- Class ID\
- Tracking ID\
- Confidence score

## 6.3 Contact Point Calculation

    contact_point = (center_x, bottom_y)

Used to check lane entry.

## 6.4 Violation Logic

    if motorcycle AND point_inside_polygon:
        mark as violation

Violations logged once per Track ID.

------------------------------------------------------------------------

# 7. Output Format

## 7.1 Video Output

-   Bounding boxes color-coded:
    -   Green: Normal vehicles
    -   Red: Motorcycles violating
    -   Blue: Non-target vehicles

## 7.2 JSON Output

Example:

``` json
[
    {
        "timestamp": "00:00:14.52",
        "vehicle_id": "moto_12",
        "type": "Lane Violation"
    }
]
```

------------------------------------------------------------------------

# 8. Evaluation Metrics (Optional)

If expanding into research or production:

-   mAP for detection accuracy\
-   IDF1 / MOTA for tracking stability\
-   Violation detection precision/recall\
-   FPS for real-time performance

------------------------------------------------------------------------

# 9. Results (Example Section)

Add screenshots or text describing performance:

-   Detection rate: 95% on daytime video\
-   Tracking stability: Good on low-traffic scenes\
-   Violation detection: Correctly flags illegal motorcycle movements

------------------------------------------------------------------------

# 10. Limitations

-   Depends heavily on polygon accuracy\
-   Performance drops in:
    -   Nighttime scenes\
    -   Rain / fog\
    -   Heavy occlusion\
    -   Extreme camera vibration\
-   No lane segmentation (polygon only)

------------------------------------------------------------------------

# 11. Future Improvements

-   Replace polygon with deep-learning lane segmentation\
-   Multi-camera fusion\
-   Edge deployment using TensorRT\
-   Automatic lane boundary generation\
-   Integration with real-time dashboards

------------------------------------------------------------------------

# 12. Conclusion

The Lane Violation Detection System demonstrates an effective pipeline
for detecting motorcycle violations using YOLOv10, ByteTrack, and
polygon-based lane modeling. It produces both visual and JSON logs
suitable for traffic monitoring use cases.

------------------------------------------------------------------------

# 13. References

-   YOLOv10 Model -- Ultralytics\
-   ByteTrack Algorithm\
-   OpenCV Documentation\
-   AICity Challenge Datasets
