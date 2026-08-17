"""
Central configuration for the Intelligent Traffic Management System.

Supports:
- Uploaded traffic videos
- Live phone/IP camera
- ONNX vehicle detection
- Traffic density
- Lane analysis
"""

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "best.onnx"

UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)


# ============================================================
# UPLOAD VALIDATION
# ============================================================

ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv"
}

MAX_CONTENT_LENGTH_MB = 300

MAX_CONTENT_LENGTH = (
    MAX_CONTENT_LENGTH_MB * 1024 * 1024
)


# ============================================================
# ONNX MODEL / INFERENCE
# ============================================================

INPUT_SIZE = 640

CONFIDENCE_THRESHOLD = 0.35

NMS_IOU_THRESHOLD = 0.45


# ============================================================
# ONNX RUNTIME
# ============================================================

# CUDA GPU first.
# CPU fallback if CUDA is unavailable.

EXECUTION_PROVIDERS_PRIORITY = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]


# ============================================================
# LIVE PHONE / IP CAMERA
# ============================================================

# IMPORTANT:
# Replace this with the URL shown by your IP Webcam app.
#
# Example:
# http://10.47.168.233:8080/video

PHONE_CAMERA_URL = (
    "http://10.47.168.233:8080/video"
)


# ------------------------------------------------------------
# Live camera settings
# ------------------------------------------------------------

LIVE_CAMERA_ENABLED = True

# Target live processing FPS.
# This does NOT change the phone camera FPS directly.
LIVE_FPS = 20

# Confidence threshold specifically for live detection.
LIVE_CONFIDENCE_THRESHOLD = 0.35

# JPEG quality sent from Flask to browser.
# 60-85 is normally a good range.
LIVE_JPEG_QUALITY = 80

# OpenCV camera buffer.
# Lower value = lower latency.
LIVE_BUFFER_SIZE = 1


# ------------------------------------------------------------
# Live processing resolution
# ------------------------------------------------------------

# Your phone is sending:
#
# 1920 x 1080
#
# We don't necessarily want to run AI inference on the
# complete 1920x1080 frame.
#
# The detector can resize internally to INPUT_SIZE=640.

LIVE_PROCESS_WIDTH = 1280
LIVE_PROCESS_HEIGHT = 720


# ------------------------------------------------------------
# Live stream reconnect settings
# ------------------------------------------------------------

LIVE_RECONNECT_ENABLED = True

LIVE_RECONNECT_DELAY = 2

LIVE_MAX_RECONNECT_ATTEMPTS = 5


# ============================================================
# FALLBACK CLASS NAMES
# ============================================================

# Used only if ONNX model does not contain class metadata.

FALLBACK_CLASS_NAMES = {

    0: "Bus",

    1: "Car",

    2: "Motorcycle",

    3: "Pickup",

    4: "Truck",
}


# ============================================================
# VEHICLE CLASSES
# ============================================================

VEHICLE_CLASSES = [

    "Bus",

    "Car",

    "Motorcycle",

    "Pickup",

    "Truck",
]


# ============================================================
# TRAFFIC DENSITY
# ============================================================

DENSITY_THRESHOLDS = {

    "LOW": (
        0,
        3
    ),

    "MODERATE": (
        4,
        6
    ),

    "HIGH": (
        7,
        10
    ),

    "SEVERE": (
        11,
        float("inf")
    ),
}


# ============================================================
# LANE POLYGONS
# ============================================================

# Coordinates are fractions of frame dimensions.
#
# This makes them automatically scale with:
#
# 1920x1080
# 1280x720
# 640x640
# etc.
#
# IMPORTANT:
# These are generic starting regions.
# For your actual phone-camera view, adjust these according
# to the road/camera position.

LANE_POLYGONS = {

    "LEFT_LANE": [

        (0.00, 0.35),

        (0.50, 0.35),

        (0.50, 1.00),

        (0.00, 1.00),
    ],


    "RIGHT_LANE": [

        (0.50, 0.35),

        (1.00, 0.35),

        (1.00, 1.00),

        (0.50, 1.00),
    ],
}


# ============================================================
# DRAWING / OVERLAY
# ============================================================

BOX_COLOR = (
    46,
    204,
    113
)

BOX_THICKNESS = 2

FONT = "FONT_HERSHEY_SIMPLEX"

FONT_SCALE = 0.5

FONT_THICKNESS = 1


# ------------------------------------------------------------
# Traffic overlay
# ------------------------------------------------------------

OVERLAY_BG_COLOR = (
    20,
    20,
    20
)

OVERLAY_TEXT_COLOR = (
    255,
    255,
    255
)


# ------------------------------------------------------------
# Lane visualization
# ------------------------------------------------------------

LANE_LINE_COLOR = (
    255,
    200,
    0
)

LANE_LINE_THICKNESS = 2


# ============================================================
# LIVE CAMERA OVERLAY
# ============================================================

LIVE_OVERLAY_ENABLED = True

LIVE_SHOW_LANES = True

LIVE_SHOW_DETECTIONS = True

LIVE_SHOW_CONFIDENCE = True

LIVE_SHOW_TRAFFIC_DENSITY = True

LIVE_SHOW_VEHICLE_COUNT = True

LIVE_SHOW_LIVE_INDICATOR = True


# ============================================================
# LIVE STREAM DISPLAY
# ============================================================

LIVE_STREAM_WIDTH = 1280

LIVE_STREAM_HEIGHT = 720

LIVE_STREAM_JPEG_QUALITY = 80


# ============================================================
# TRAFFIC SIGNAL CONTROL
# ============================================================

# Final per-lane signal rule (see traffic_analysis.signal_for_density):
#   LOW density                  -> RED    (lane doesn't need servicing)
#   MODERATE / HIGH / SEVERE     -> GREEN  (lane has traffic to clear)
#
# When a lane's final signal is GREEN, this suggests how long that
# lane's light should stay green, based on how busy it is. Purely a
# suggestion shown on the dashboard - tune freely per intersection.
SIGNAL_GREEN_DURATION_SECONDS = {
    "LOW": 0,
    "MODERATE": 20,
    "HIGH": 30,
    "SEVERE": 45,
}

# BGR colors (OpenCV convention) for drawing the signal-light circles
# directly onto video frames.
SIGNAL_LIGHT_COLORS = {
    "RED": (46, 46, 220),
    "GREEN": (90, 190, 60),
}


# ============================================================
# TRAFFIC SCORE (composite congestion score, 0.0-1.0 per lane)
# ============================================================
#
#   S = 0.35*(MaxVehicles/Capacity)
#     + 0.25*(AvgVehicles/Capacity)
#     + 0.25*(Utilization)
#     + 0.15*(DensityScore)
#
# Definitions used here (documented since the raw formula doesn't
# pin these down numerically):
#   - Capacity        : max vehicles a lane can physically hold at
#                        once, per LANE_CAPACITY below. MaxVehicles/
#                        Capacity and AvgVehicles/Capacity are each
#                        clamped to 1.0 so a single lane can't exceed
#                        its own contribution to S even if a burst of
#                        detections briefly exceeds the configured
#                        capacity.
#   - Utilization      : fraction of frames (0.0-1.0) in which this
#                         lane had at least one vehicle present -
#                         "how much of the time is this lane in use",
#                         independent of how many vehicles at once.
#   - DensityScore     : the lane's existing LOW/MODERATE/HIGH/SEVERE
#                         label (from DENSITY_THRESHOLDS, unchanged)
#                         mapped to a 0-1 number via DENSITY_SCORE_MAP.
#
# The four weights sum to 1.0, so S naturally lands in [0, 1].

# Max vehicles a lane can hold - used to normalize the Max/Avg terms.
# Override per lane name if your lanes differ in size; unlisted lanes
# fall back to DEFAULT_LANE_CAPACITY.
LANE_CAPACITY = {
    "LEFT_LANE": 10,
    "RIGHT_LANE": 10,
}
DEFAULT_LANE_CAPACITY = 10

TRAFFIC_SCORE_WEIGHTS = {
    "max": 0.35,
    "avg": 0.25,
    "utilization": 0.25,
    "density": 0.15,
}

# Maps a density label to a 0-1 score for use inside the composite
# formula above. Does NOT change DENSITY_THRESHOLDS or how a lane
# gets classified as LOW/MODERATE/HIGH/SEVERE in the first place.
DENSITY_SCORE_MAP = {
    "LOW": 0.25,
    "MODERATE": 0.50,
    "HIGH": 0.75,
    "SEVERE": 1.00,
}

# Final composite score -> a human-readable congestion label, purely
# for display (separate from the RED/GREEN signal above).
TRAFFIC_SCORE_THRESHOLDS = {
    "LOW": (0.00, 0.25),
    "MODERATE": (0.25, 0.50),
    "HIGH": (0.50, 0.75),
    "SEVERE": (0.75, 1.01),   # 1.01 so a perfect 1.0 score still lands in SEVERE
}


# ============================================================
# CHART
# ============================================================

MAX_CHART_POINTS = 150