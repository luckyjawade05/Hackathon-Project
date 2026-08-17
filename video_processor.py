"""
video_processor.py
Drives the frame-by-frame pipeline: read frame -> detect -> analyze ->
draw overlay -> write frame. Never loads the whole video into memory;
uses cv2.VideoCapture to stream frames in, and a direct FFmpeg
subprocess pipe to stream annotated frames out as real H.264
(see _FrameWriter below).
"""
import subprocess
import threading
from pathlib import Path

import cv2

import config
from traffic_analysis import TrafficStatsTracker, build_lane_polygons

FONT = getattr(cv2, config.FONT)

try:
    import imageio_ffmpeg
    _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _FFMPEG_EXE = None


class VideoProcessingError(Exception):
    """Raised for any unrecoverable error while processing a video."""


class _FrameWriter:
    """Writes annotated BGR frames to a browser-playable H.264 MP4.

    Standard pip `opencv-python` wheels are built without a licensed
    H.264 encoder, so cv2.VideoWriter silently falls back to codecs
    (e.g. mp4v) that OpenCV can read back but that browsers refuse to
    play inline. This pipes raw BGR frames directly into a static
    FFmpeg binary (via imageio-ffmpeg, no system-wide FFmpeg install
    needed) over stdin - a plain subprocess pipe, not a Python
    wrapper library, so there's no version/plugin selection to break
    across environments. Falls back to cv2.VideoWriter only if the
    FFmpeg binary itself is unavailable.

    stderr is drained continuously on a background thread: on Windows
    especially, if FFmpeg's stderr pipe fills up while we're still
    writing frames to stdin, FFmpeg blocks trying to write stderr and
    the whole pipeline deadlocks. Reading stderr as it arrives avoids
    that regardless of how much FFmpeg logs.
    """

    def __init__(self, output_path, fps, width, height):
        self.width = width
        self.height = height
        self._backend = None
        self._proc = None
        self._writer = None
        self._stderr_chunks = []
        self._stderr_thread = None

        if _FFMPEG_EXE is not None:
            cmd = [
                _FFMPEG_EXE, "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path),
            ]
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                self._backend = "ffmpeg"
                self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
                self._stderr_thread.start()
            except Exception as exc:
                print(f"[video_processor] FFmpeg pipe unavailable, "
                      f"falling back to cv2.VideoWriter: {exc}")
                self._proc = None

        if self._proc is None:
            self._writer = self._open_cv2_writer(output_path, fps, width, height)
            self._backend = "cv2"

        if self._backend == "cv2" and self._writer is None:
            raise VideoProcessingError(
                "Could not create the output video file (no working video writer found)."
            )

    def _drain_stderr(self):
        try:
            for chunk in iter(lambda: self._proc.stderr.read(4096), b""):
                self._stderr_chunks.append(chunk)
        except Exception:
            pass

    @staticmethod
    def _open_cv2_writer(output_path, fps, width, height):
        for fourcc_str in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if writer.isOpened():
                return writer
            writer.release()
        return None

    def write(self, bgr_frame):
        if self._backend == "ffmpeg":
            if self._proc.poll() is not None:
                raise VideoProcessingError(
                    "Video encoder exited unexpectedly while processing: "
                    + b"".join(self._stderr_chunks).decode(errors="ignore")[:300]
                )
            # Defensive: guarantee every frame matches the declared raw
            # frame size, even if an individual decoded frame's actual
            # shape differs from what CAP_PROP reported (can happen with
            # rotation metadata or variable-resolution streams).
            if bgr_frame.shape[1] != self.width or bgr_frame.shape[0] != self.height:
                bgr_frame = cv2.resize(bgr_frame, (self.width, self.height))
            self._proc.stdin.write(bgr_frame.tobytes())
        else:
            self._writer.write(bgr_frame)

    def release(self):
        if self._backend == "ffmpeg":
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            self._proc.wait()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=5)
            if self._proc.returncode != 0:
                raise VideoProcessingError(
                    "Video encoding failed: "
                    + b"".join(self._stderr_chunks).decode(errors="ignore")[:300]
                )
        else:
            self._writer.release()


def _open_writer(output_path, fps, width, height):
    try:
        return _FrameWriter(output_path, fps, width, height)
    except VideoProcessingError:
        return None


def _draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), config.BOX_COLOR, config.BOX_THICKNESS)

        (tw, th), _ = cv2.getTextSize(label, FONT, config.FONT_SCALE, config.FONT_THICKNESS)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), config.BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, max(12, y1 - 4)), FONT,
                    config.FONT_SCALE, (0, 0, 0), config.FONT_THICKNESS, cv2.LINE_AA)


def _draw_lanes(frame, lane_polygons):
    for lane_name, polygon in lane_polygons.items():
        cv2.polylines(frame, [polygon], isClosed=True,
                       color=config.LANE_LINE_COLOR, thickness=config.LANE_LINE_THICKNESS)
        x, y = polygon[0]
        cv2.putText(frame, lane_name.replace("_", " "), (int(x) + 5, int(y) + 20),
                    FONT, 0.6, config.LANE_LINE_COLOR, 2, cv2.LINE_AA)


def _draw_stats_panel(frame, frame_stats):
    lines = [
        "INTELLIGENT TRAFFIC MANAGEMENT SYSTEM",
        f"Total Vehicles: {frame_stats['vehicle_count']}",
        f"Traffic Density: {frame_stats['overall_density']}",
    ]
    lane_signal_rows = []   # (line_index, lane) - for the RED/GREEN circle
    for lane, count in frame_stats["lane_counts"].items():
        density = frame_stats["lane_densities"][lane]
        signal = frame_stats["lane_signals"][lane]["signal"]
        score = frame_stats["lane_scores"][lane]
        lines.append(f"{lane.replace('_', ' ')}: {count} ({density})  SIGNAL: {signal}")
        lane_signal_rows.append((len(lines) - 1, lane))
        lines.append(f"   Traffic Score: {score['score_pct']}% ({score['score_label']})")

    panel_w = 420
    panel_h = 22 * len(lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), config.OVERLAY_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 32
    line_ys = {}
    for i, line in enumerate(lines):
        weight = 2 if i == 0 else 1
        font_scale = 0.55 if i not in [row[0] + 1 for row in lane_signal_rows] else 0.48
        cv2.putText(frame, line, (20, y), FONT, font_scale, config.OVERLAY_TEXT_COLOR, weight, cv2.LINE_AA)
        line_ys[i] = y
        y += 22

    # Small filled circle (red/green) at the end of each lane's line,
    # giving an at-a-glance signal indicator directly on the video.
    for line_idx, lane in lane_signal_rows:
        signal = frame_stats["lane_signals"][lane]["signal"]
        color = config.SIGNAL_LIGHT_COLORS.get(signal, (200, 200, 200))
        cy = line_ys[line_idx] - 5
        cv2.circle(frame, (panel_w - 10, cy), 7, color, -1)
        cv2.circle(frame, (panel_w - 10, cy), 7, (255, 255, 255), 1)


def _validate_output(output_path, expected_min_frames=1):
    """Confirm the written file is actually a readable video before we
    tell the browser it's ready - catches silent encoder failures that
    still exit with code 0 but produce an unplayable/truncated file."""
    output_path = Path(output_path)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise VideoProcessingError(
            "Video processing finished but no output file was created."
        )
    check_cap = cv2.VideoCapture(str(output_path))
    try:
        if not check_cap.isOpened():
            raise VideoProcessingError(
                "The processed video file could not be verified as playable."
            )
        ret, _ = check_cap.read()
        if not ret:
            raise VideoProcessingError(
                "The processed video file was created but contains no readable frames."
            )
    finally:
        check_cap.release()


def process_video(input_path, output_path, detector, progress_callback=None):
    """Process input_path frame-by-frame and write the annotated result
    to output_path. Returns a summary stats dict on success.
    Raises VideoProcessingError on any failure."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise VideoProcessingError(
            "The uploaded video could not be opened. It may be corrupted "
            "or in an unsupported codec."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:   # NaN / 0 guard
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame now and derive width/height from its actual
    # decoded shape rather than trusting CAP_PROP_FRAME_WIDTH/HEIGHT,
    # which can disagree with reality for rotated or variable-resolution
    # streams. This is what we pass to the encoder, so it must match
    # exactly or the output container comes out corrupt.
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise VideoProcessingError("The uploaded video has no readable frames.")

    height, width = first_frame.shape[:2]
    if width <= 0 or height <= 0:
        cap.release()
        raise VideoProcessingError("The uploaded video has no readable frames.")

    writer = _open_writer(output_path, fps, width, height)
    if writer is None:
        cap.release()
        raise VideoProcessingError(
            "Could not create the output video file (no working video codec found)."
        )

    lane_polygons = build_lane_polygons(width, height)
    tracker = TrafficStatsTracker(lane_polygons)

    frame_index = 0
    pending_frame = first_frame
    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            try:
                detections = detector.detect(frame)
            except RuntimeError as exc:
                raise VideoProcessingError(str(exc)) from exc

            frame_stats = tracker.update(detections, frame_index)

            _draw_lanes(frame, lane_polygons)
            _draw_detections(frame, detections)
            _draw_stats_panel(frame, frame_stats)

            writer.write(frame)
            frame_index += 1

            if progress_callback and total_frames > 0:
                progress_callback(frame_index, total_frames)

        if frame_index == 0:
            raise VideoProcessingError("The uploaded video contains no frames to process.")

    finally:
        cap.release()
        writer.release()

    _validate_output(output_path)

    summary = tracker.summary()
    summary["fps"] = round(fps, 2)
    summary["resolution"] = f"{width}x{height}"
    return summary
