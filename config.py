"""
Configuration file for smart parking system.

Parking violation thresholds and weights.
"""

# Alignment thresholds
DELTA = 0.5  # Alignment score threshold for improper parking
EPSILON = 0.3  # IoU threshold for improper parking

# Temporal verification
CONFIRMATION_THRESHOLD = 3  # Frames required to confirm violation
WINDOW_SIZE = 10  # Historical frame window

# Occupancy detection
ALIGNMENT_THRESHOLD = 0.5  # Alignment score for proper parking
IOU_THRESHOLD = 0.3  # Minimum IoU for occupancy detection
CONFIDENCE_THRESHOLD = 0.5  # Minimum detector confidence

# Penalty engine weights (α, β, γ)
# Penalty = α·Duration + β·Severity + γ·Repetition
ALPHA = 0.3  # Duration weight (fairness)
BETA = 0.4  # Severity weight (spatial impact)
GAMMA = 0.3  # Repetition weight (behavioral pattern)
MAX_PENALTY = 100.0  # Maximum penalty score
MAX_DURATION_BASELINE = 3600.0  # Reference duration (seconds, 1 hour)

# Allocation strategy weights
SPATIAL_WEIGHT = 0.5  # Alignment score weight
TEMPORAL_WEIGHT = 0.3  # Temporal verification weight
CONFIDENCE_WEIGHT = 0.2  # Detector confidence weight
VIOLATION_PENALTY = 0.7  # Penalty multiplier for violations

# YOLO detector
YOLO_MODEL_TYPE = "yolov8"  # "yolov7" or "yolov8"
YOLO_CONFIDENCE_THRESHOLD = 0.5
VEHICLE_CLASSES = [2, 5, 7, 8]  # COCO: car, bus, truck variants
