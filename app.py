"""
app.py

Flask entry point for the Intelligent Traffic Management System.

Features:
- Uploaded video processing
- Live phone/IP camera processing
- ONNX vehicle detection
- Traffic density analysis
- Lane analysis
- Server-rendered HTML
- MJPEG live video streaming

NO jsonify()
NO JSON API routes
"""

import time
import uuid
import threading
from pathlib import Path

import cv2

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    Response,
)

from werkzeug.utils import secure_filename

import config

from detector import (
    VehicleDetector,
    ModelLoadError,
)

from video_processor import (
    process_video,
    VideoProcessingError,
)

from traffic_analysis import (
    TrafficStatsTracker,
    build_lane_polygons,
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    config.MAX_CONTENT_LENGTH
)

app.secret_key = "itms-dev-secret-key"


# ============================================================
# LIVE CAMERA SHARED STATE
# ============================================================
# generate_live_frames() runs the phone-camera loop and updates this
# on every frame; /live_stats reads it to render an HTML fragment the
# frontend polls, WITHOUT a JSON API - just Flask-rendered HTML, same
# rule as the rest of the app.

live_state_lock = threading.Lock()

live_state = {
    "active": False,
    "frame_stats": None,
    "resolution": None,
    "frame_index": 0,
}


# ============================================================
# LOAD ONNX MODEL ONCE
# ============================================================

detector = None

model_load_error = None

try:

    detector = VehicleDetector(
        config.MODEL_PATH
    )

    print(
        "[app] ONNX model loaded successfully."
    )

except ModelLoadError as exc:

    model_load_error = str(exc)

    print(
        f"[app] {model_load_error}"
    )


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "."
        in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in config.ALLOWED_EXTENSIONS
    )


# ============================================================
# DEFAULT TEMPLATE CONTEXT
# ============================================================

def default_context():

    return {

        "error": None,

        "results": None,

        "original_video_url": None,

        "processed_video_url": None,

        "processed_filename": None,

        "class_names": sorted(
            config.VEHICLE_CLASSES
        ),

        "model_ready": detector is not None,

        "model_error": model_load_error,
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def index():

    return render_template(
        "index.html",
        **default_context()
    )


# ============================================================
# UPLOAD + PROCESS VIDEO
# ============================================================

@app.route(
    "/process",
    methods=["POST"]
)
def process():

    ctx = default_context()

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if detector is None:

        ctx["error"] = (
            "The AI model is not available: "
            f"{model_load_error or 'unknown error loading model.'}"
        )

        return render_template(
            "index.html",
            **ctx
        )

    # --------------------------------------------------------
    # Get uploaded file
    # --------------------------------------------------------

    upload = request.files.get(
        "video"
    )

    if (
        upload is None
        or upload.filename == ""
    ):

        ctx["error"] = (
            "Please choose a video file "
            "before uploading."
        )

        return render_template(
            "index.html",
            **ctx
        )

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    if not allowed_file(
        upload.filename
    ):

        ctx["error"] = (
            "Unsupported file format. "
            "Allowed formats: "
            f"{', '.join(sorted(config.ALLOWED_EXTENSIONS)).upper()}"
        )

        return render_template(
            "index.html",
            **ctx
        )

    # --------------------------------------------------------
    # Create unique filenames
    # --------------------------------------------------------

    original_name = secure_filename(
        upload.filename
    )

    ext = original_name.rsplit(
        ".",
        1
    )[1].lower()

    unique_id = uuid.uuid4().hex[:10]

    input_filename = (
        f"{unique_id}_input.{ext}"
    )

    output_filename = (
        f"{unique_id}_processed.mp4"
    )

    input_path = (
        config.UPLOAD_FOLDER
        / input_filename
    )

    output_path = (
        config.OUTPUT_FOLDER
        / output_filename
    )

    # --------------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------------

    try:

        upload.save(
            str(input_path)
        )

    except Exception as exc:

        ctx["error"] = (
            "Failed to save the uploaded file. "
            "Please try again."
        )

        print(
            f"[app] Upload save error: {exc}"
        )

        return render_template(
            "index.html",
            **ctx
        )

    # --------------------------------------------------------
    # Process video
    # --------------------------------------------------------

    start_time = time.time()

    try:

        summary = process_video(
            input_path,
            output_path,
            detector
        )

    except VideoProcessingError as exc:

        ctx["error"] = str(exc)

        _cleanup(
            input_path,
            output_path
        )

        return render_template(
            "index.html",
            **ctx
        )

    except Exception as exc:

        print(
            f"[app] Unexpected processing error: {exc}"
        )

        ctx["error"] = (
            "An unexpected error occurred "
            "while processing the video."
        )

        _cleanup(
            input_path,
            output_path
        )

        return render_template(
            "index.html",
            **ctx
        )

    # --------------------------------------------------------
    # Processing time
    # --------------------------------------------------------

    summary["processing_seconds"] = round(
        time.time() - start_time,
        1
    )

    # --------------------------------------------------------
    # Send result to template
    # --------------------------------------------------------

    ctx["results"] = summary

    ctx["original_video_url"] = (
        f"/media/uploads/{input_filename}"
    )

    ctx["processed_video_url"] = (
        f"/media/outputs/{output_filename}"
    )

    ctx["processed_filename"] = (
        output_filename
    )

    return render_template(
        "index.html",
        **ctx
    )


# ============================================================
# CLEANUP
# ============================================================

def _cleanup(*paths):

    for path in paths:

        try:

            Path(path).unlink(
                missing_ok=True
            )

        except Exception:

            pass


# ============================================================
# MEDIA SERVING
# ============================================================

@app.route(
    "/media/uploads/<path:filename>"
)
def media_uploads(filename):

    safe_name = secure_filename(
        filename
    )

    return send_from_directory(
        config.UPLOAD_FOLDER,
        safe_name
    )


@app.route(
    "/media/outputs/<path:filename>"
)
def media_outputs(filename):

    safe_name = secure_filename(
        filename
    )

    return send_from_directory(
        config.OUTPUT_FOLDER,
        safe_name
    )


# ============================================================
# DOWNLOAD PROCESSED VIDEO
# ============================================================

@app.route(
    "/download/<path:filename>"
)
def download(filename):

    safe_name = secure_filename(
        filename
    )

    return send_from_directory(
        config.OUTPUT_FOLDER,
        safe_name,
        as_attachment=True
    )


# ============================================================
# LIVE PHONE CAMERA
# ============================================================

def generate_live_frames():

    """
    Reads frames continuously from the phone/IP camera.

    Pipeline:

        Phone Camera
             ↓
        OpenCV
             ↓
        ONNX Detector
             ↓
        Traffic Analysis
             ↓
        Draw Bounding Boxes
             ↓
        Draw Lane Information
             ↓
        JPEG
             ↓
        Flask MJPEG Stream
    """

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if detector is None:

        print(
            "[LIVE] Model is not available."
        )

        return

    # --------------------------------------------------------
    # Phone camera URL
    # --------------------------------------------------------

    camera_url = (
        config.PHONE_CAMERA_URL
    )

    print("\n")
    print("=" * 60)
    print("STARTING LIVE PHONE CAMERA")
    print("=" * 60)

    print(
        "Camera URL:",
        camera_url
    )

    # --------------------------------------------------------
    # Open phone camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        camera_url
    )

    # Reduce buffering
    try:

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            config.LIVE_BUFFER_SIZE
        )

    except Exception:

        pass

    # --------------------------------------------------------
    # Check connection
    # --------------------------------------------------------

    if not cap.isOpened():

        print(
            "[LIVE] ERROR: "
            "Could not connect to phone camera."
        )

        return

    print(
        "[LIVE] Phone camera connected successfully."
    )

    # --------------------------------------------------------
    # Read first frame
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print(
            "[LIVE] ERROR: "
            "Could not read first frame."
        )

        cap.release()

        return

    # --------------------------------------------------------
    # Get resolution
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    print(
        f"[LIVE] Resolution: "
        f"{width}x{height}"
    )

    # --------------------------------------------------------
    # Build lane polygons
    # --------------------------------------------------------

    lane_polygons = build_lane_polygons(
        width,
        height
    )

    # --------------------------------------------------------
    # Traffic tracker
    # --------------------------------------------------------

    tracker = TrafficStatsTracker(
        lane_polygons
    )

    frame_index = 0

    # Reset shared state for this new stream session.
    with live_state_lock:
        live_state["active"] = True
        live_state["frame_stats"] = None
        live_state["resolution"] = f"{width}x{height}"
        live_state["frame_index"] = 0

    # ========================================================
    # FRAME LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret:

                print(
                    "[LIVE] Frame read failed."
                )

                break

            # ------------------------------------------------
            # ONNX DETECTION
            # ------------------------------------------------

            try:

                detections = detector.detect(
                    frame
                )

            except Exception as exc:

                print(
                    "[LIVE] Detection error:",
                    exc
                )

                continue

            # ------------------------------------------------
            # Traffic analysis
            # ------------------------------------------------

            frame_stats = tracker.update(
                detections,
                frame_index
            )

            with live_state_lock:
                live_state["frame_stats"] = frame_stats
                live_state["frame_index"] = frame_index + 1

            # =================================================
            # DRAW LANES
            # =================================================

            if getattr(
                config,
                "LIVE_SHOW_LANES",
                True
            ):

                for (
                    lane_name,
                    polygon
                ) in lane_polygons.items():

                    cv2.polylines(
                        frame,
                        [polygon],
                        isClosed=True,
                        color=config.LANE_LINE_COLOR,
                        thickness=config.LANE_LINE_THICKNESS
                    )

                    x, y = polygon[0]

                    cv2.putText(
                        frame,
                        lane_name.replace(
                            "_",
                            " "
                        ),
                        (
                            int(x) + 5,
                            int(y) + 25
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        config.LANE_LINE_COLOR,
                        2,
                        cv2.LINE_AA
                    )

            # =================================================
            # DRAW DETECTIONS
            # =================================================

            if getattr(
                config,
                "LIVE_SHOW_DETECTIONS",
                True
            ):

                for det in detections:

                    x1, y1, x2, y2 = (
                        det["bbox"]
                    )

                    class_name = (
                        det["class_name"]
                    )

                    confidence = (
                        det["confidence"]
                    )

                    # ------------------------------------------------
                    # Label
                    # ------------------------------------------------

                    if getattr(
                        config,
                        "LIVE_SHOW_CONFIDENCE",
                        True
                    ):

                        label = (
                            f"{class_name} "
                            f"{confidence:.2f}"
                        )

                    else:

                        label = class_name

                    # ------------------------------------------------
                    # Bounding box
                    # ------------------------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        config.BOX_COLOR,
                        config.BOX_THICKNESS
                    )

                    # ------------------------------------------------
                    # Label size
                    # ------------------------------------------------

                    (
                        text_width,
                        text_height
                    ), baseline = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        2
                    )

                    label_top = max(
                        0,
                        y1 - text_height - 12
                    )

                    label_bottom = (
                        y1
                    )

                    # ------------------------------------------------
                    # Label background
                    # ------------------------------------------------

                    cv2.rectangle(
                        frame,
                        (
                            x1,
                            label_top
                        ),
                        (
                            x1 + text_width + 10,
                            label_bottom
                        ),
                        config.BOX_COLOR,
                        -1
                    )

                    # ------------------------------------------------
                    # Label text
                    # ------------------------------------------------

                    cv2.putText(
                        frame,
                        label,
                        (
                            x1 + 5,
                            max(
                                text_height + 2,
                                y1 - 5
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA
                    )

            # =================================================
            # TRAFFIC INFORMATION PANEL
            # =================================================

            lines = []

            # ------------------------------------------------
            # Title
            # ------------------------------------------------

            lines.append(
                "LIVE TRAFFIC MANAGEMENT"
            )

            # ------------------------------------------------
            # Vehicle count
            # ------------------------------------------------

            if getattr(
                config,
                "LIVE_SHOW_VEHICLE_COUNT",
                True
            ):

                lines.append(
                    f"Vehicles: "
                    f"{frame_stats['vehicle_count']}"
                )

            # ------------------------------------------------
            # Overall density
            # ------------------------------------------------

            if getattr(
                config,
                "LIVE_SHOW_TRAFFIC_DENSITY",
                True
            ):

                lines.append(
                    f"Traffic Density: "
                    f"{frame_stats['overall_density']}"
                )

# ------------------------------------------------
            # Lane information
            # ------------------------------------------------

            lane_signal_rows = []

            for (
                lane,
                count
            ) in frame_stats[
                "lane_counts"
            ].items():

                density = (
                    frame_stats[
                        "lane_densities"
                    ][lane]
                )

                signal = (
                    frame_stats[
                        "lane_signals"
                    ][lane]["signal"]
                )

                score = (
                    frame_stats[
                        "lane_scores"
                    ][lane]
                )

                lines.append(
                    f"{lane.replace('_', ' ')}: "
                    f"{count} "
                    f"({density})  SIGNAL: {signal}"
                )

                lane_signal_rows.append(
                    (len(lines) - 1, lane)
                )

                lines.append(
                    f"   Traffic Score: "
                    f"{score['score_pct']}% "
                    f"({score['score_label']})"
                )

            # =================================================
            # PANEL
            # =================================================

            panel_width = 500

            panel_height = (
                30 * len(lines)
                + 25
            )

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (10, 10),
                (
                    panel_width,
                    panel_height
                ),
                config.OVERLAY_BG_COLOR,
                -1
            )

            cv2.addWeighted(
                overlay,
                0.70,
                frame,
                0.30,
                0,
                frame
            )

            # =================================================
            # PANEL TEXT
            # =================================================

            y = 40

            line_ys = {}

            score_line_indices = {
                idx for idx, _ in lane_signal_rows
                for idx in [idx + 1]
            }

            for index, line in enumerate(
                lines
            ):

                thickness = (
                    2
                    if index == 0
                    else 1
                )

                font_scale = (
                    0.55
                    if index in score_line_indices
                    else 0.65
                )

                cv2.putText(
                    frame,
                    line,
                    (
                        20,
                        y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    config.OVERLAY_TEXT_COLOR,
                    thickness,
                    cv2.LINE_AA
                )

                line_ys[index] = y

                y += 30
            # =================================================
            # SIGNAL LIGHT INDICATORS
            # =================================================

            for line_idx, lane in lane_signal_rows:

                signal = (
                    frame_stats[
                        "lane_signals"
                    ][lane]["signal"]
                )

                color = config.SIGNAL_LIGHT_COLORS.get(
                    signal,
                    (200, 200, 200)
                )

                cy = line_ys[line_idx] - 7

                cv2.circle(
                    frame,
                    (panel_width - 15, cy),
                    8,
                    color,
                    -1
                )

                cv2.circle(
                    frame,
                    (panel_width - 15, cy),
                    8,
                    (255, 255, 255),
                    1
                )

            # =================================================
            # LIVE INDICATOR
            # =================================================

            if getattr(
                config,
                "LIVE_SHOW_LIVE_INDICATOR",
                True
            ):

                # Red dot

                cv2.circle(
                    frame,
                    (
                        width - 45,
                        35
                    ),
                    10,
                    (0, 0, 255),
                    -1
                )

                # LIVE text

                cv2.putText(
                    frame,
                    "LIVE",
                    (
                        width - 105,
                        42
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # =================================================
            # JPEG ENCODING
            # =================================================

            jpeg_quality = getattr(
                config,
                "LIVE_JPEG_QUALITY",
                80
            )

            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    jpeg_quality
                ]
            )

            if not success:

                continue

            frame_bytes = (
                buffer.tobytes()
            )

            # =================================================
            # SEND FRAME TO BROWSER
            # =================================================

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(
                    len(frame_bytes)
                ).encode()
                + b"\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

            frame_index += 1

    except GeneratorExit:

        print(
            "[LIVE] Browser disconnected."
        )

    except Exception as exc:

        print(
            "[LIVE] Stream error:",
            exc
        )

    finally:

        with live_state_lock:
            live_state["active"] = False

        cap.release()

        print(
            "[LIVE] Phone camera released."
        )

        print(
            "=" * 60
        )


# ============================================================
# LIVE CAMERA ROUTE
# ============================================================

@app.route(
    "/live",
    methods=["GET"]
)
def live():

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if detector is None:

        return (
            "ONNX model is not available.",
            503
        )

    # --------------------------------------------------------
    # Stream live frames
    # --------------------------------------------------------

    return Response(
        generate_live_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    )


# ============================================================
# LIVE CAMERA STATS (HTML FRAGMENT, POLLED BY THE FRONTEND)
# ============================================================
# Returns a small rendered HTML snippet - the same traffic-signal
# widgets and stats used for uploaded videos, but built from whatever
# frame the live phone-camera loop most recently processed. This is
# NOT a JSON API: it's a normal Flask template render, returned as
# plain HTML text, which the frontend swaps into the page via
# fetch().then(r => r.text()). No jsonify(), no JSON anywhere.

@app.route(
    "/live_stats",
    methods=["GET"]
)
def live_stats():

    with live_state_lock:
        state_snapshot = dict(live_state)

    return render_template(
        "_live_stats_partial.html",
        live=state_snapshot,
    )


# ============================================================
# FILE SIZE ERROR
# ============================================================

@app.errorhandler(413)
def too_large(_exc):

    ctx = default_context()

    ctx["error"] = (
        "The uploaded file is too large "
        f"(max {config.MAX_CONTENT_LENGTH_MB} MB)."
    )

    return (
        render_template(
            "index.html",
            **ctx
        ),
        413
    )


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("INTELLIGENT TRAFFIC MANAGEMENT SYSTEM")
    print("=" * 60)

    print(
        "Model:",
        config.MODEL_PATH
    )

    print(
        "Phone Camera:",
        config.PHONE_CAMERA_URL
    )

    print(
        "Website:",
        "http://127.0.0.1:5000"
    )

    print(
        "=" * 60
    )

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
        threaded=True,
    )