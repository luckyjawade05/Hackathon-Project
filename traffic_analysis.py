"""
traffic_analysis.py
Turns raw per-frame detections into traffic-management intelligence:
lane assignment, density/congestion classification, and running
statistics across the whole video (totals, peaks, averages, class
distribution, per-frame trend for the chart).
"""
import cv2
import numpy as np

import config


def classify_density(vehicle_count):
    """Map a vehicle count to a LOW/MODERATE/HIGH/SEVERE label using the
    configurable thresholds in config.DENSITY_THRESHOLDS."""
    for label, (low, high) in config.DENSITY_THRESHOLDS.items():
        if low <= vehicle_count <= high:
            return label
    return "SEVERE"


def signal_for_score(score):
    """The signal decision is driven by the composite Traffic Score S
    (see compute_traffic_score below), not directly by density:

        S >= 0.50  -> GREEN  (lane is congested enough to need priority)
        S <  0.50  -> RED    (lane is calm enough to wait)

    Applied independently per lane - each lane's score is built purely
    from that lane's own numbers (its own max/avg/utilization/density),
    so each traffic light runs on its own separate decision. Lane 1's
    signal has no effect on Lane 2's signal and vice versa.
    """
    return "GREEN" if score >= 0.5 else "RED"


def classify_traffic_score(score):
    """Map a composite traffic score (0.0-1.0) to a LOW/MODERATE/HIGH/
    SEVERE label using config.TRAFFIC_SCORE_THRESHOLDS. Purely a
    display label alongside the score."""
    for label, (low, high) in config.TRAFFIC_SCORE_THRESHOLDS.items():
        if low <= score < high:
            return label
    return "SEVERE"


def compute_traffic_score(lane_name, max_vehicles, avg_vehicles, utilization, density_label):
    """Composite per-lane congestion score:

        S = 0.35*(MaxVehicles/Capacity) + 0.25*(AvgVehicles/Capacity)
          + 0.25*(Utilization) + 0.15*(DensityScore)

    Each lane's score is computed purely from that lane's own numbers
    - no comparison against any other lane. See config.py for how
    Capacity, Utilization, and DensityScore are defined.
    """
    capacity = config.LANE_CAPACITY.get(lane_name, config.DEFAULT_LANE_CAPACITY)
    capacity = max(capacity, 1)  # guard against a misconfigured 0 capacity

    max_ratio = min(max_vehicles / capacity, 1.0)
    avg_ratio = min(avg_vehicles / capacity, 1.0)
    density_score = config.DENSITY_SCORE_MAP.get(density_label, 0.0)

    w = config.TRAFFIC_SCORE_WEIGHTS
    score = (
        w["max"] * max_ratio
        + w["avg"] * avg_ratio
        + w["utilization"] * utilization
        + w["density"] * density_score
    )
    score = round(min(max(score, 0.0), 1.0), 3)

    return {
        "score": score,
        "score_pct": round(score * 100, 1),
        "score_label": classify_traffic_score(score),
        "max_ratio": round(max_ratio, 2),
        "avg_ratio": round(avg_ratio, 2),
        "utilization": round(utilization, 2),
        "density_score": density_score,
        "capacity": capacity,
    }


def build_lane_result(lane_name, max_vehicles, avg_vehicles, utilization):
    """Compute everything for ONE lane, entirely independently of any
    other lane:

      1. density label (from this lane's own average vehicle count)
      2. composite Traffic Score S (from this lane's own max/avg/
         utilization/density)
      3. final RED/GREEN signal (from this lane's own score, via the
         >=50% rule in signal_for_score)

    Returns (signal_info, score_info) - two dicts with the exact same
    shape used everywhere else in the app (templates, video overlay,
    live overlay), so nothing downstream needs to change.
    """
    avg_vehicles_r = round(avg_vehicles, 2)
    density = classify_density(round(avg_vehicles_r))

    score_info = compute_traffic_score(
        lane_name, max_vehicles, avg_vehicles, utilization, density
    )

    signal = signal_for_score(score_info["score"])

    signal_info = {
        "avg_vehicles": avg_vehicles_r,
        "density": density,
        "signal": signal,
        "suggested_green_seconds": (
            config.SIGNAL_GREEN_DURATION_SECONDS.get(density, 0) if signal == "GREEN" else 0
        ),
    }
    return signal_info, score_info


def build_lane_polygons(frame_w, frame_h):
    """Scale the fractional lane polygons from config to actual pixel
    coordinates for this video's resolution."""
    scaled = {}
    for lane_name, points in config.LANE_POLYGONS.items():
        scaled[lane_name] = np.array(
            [(int(x * frame_w), int(y * frame_h)) for x, y in points],
            dtype=np.int32,
        )
    return scaled


def assign_lane(bbox, lane_polygons):
    """Assign a detection to a lane based on whether its bbox center
    falls inside that lane's polygon. Returns None if it falls outside
    every configured lane."""
    x1, y1, x2, y2 = bbox
    center = (float((x1 + x2) / 2), float((y1 + y2) / 2))
    for lane_name, polygon in lane_polygons.items():
        if cv2.pointPolygonTest(polygon, center, False) >= 0:
            return lane_name
    return None


class TrafficStatsTracker:
    """Accumulates traffic statistics across every processed frame."""

    def __init__(self, lane_polygons):
        self.lane_polygons = lane_polygons
        self.lane_names = list(lane_polygons.keys())

        self.frame_count = 0
        self.total_detections = 0
        self.max_vehicles_in_frame = 0
        self.sum_vehicles_per_frame = 0

        self.class_totals = {name: 0 for name in config.VEHICLE_CLASSES}
        self.lane_totals = {lane: 0 for lane in self.lane_names}
        self.lane_max_counts = {lane: 0 for lane in self.lane_names}
        self.lane_active_frames = {lane: 0 for lane in self.lane_names}

        # per-frame trend, sampled down later for the chart
        self.frame_trend = []          # list of (frame_index, vehicle_count)

    def update(self, detections, frame_index):
        self.frame_count += 1
        vehicle_count = len(detections)
        self.total_detections += vehicle_count
        self.sum_vehicles_per_frame += vehicle_count
        self.max_vehicles_in_frame = max(self.max_vehicles_in_frame, vehicle_count)

        lane_counts = {lane: 0 for lane in self.lane_names}

        for det in detections:
            cls_name = det["class_name"]
            self.class_totals[cls_name] = self.class_totals.get(cls_name, 0) + 1

            lane = assign_lane(det["bbox"], self.lane_polygons)
            det["lane"] = lane
            if lane is not None:
                lane_counts[lane] += 1
                self.lane_totals[lane] += 1

        for lane in self.lane_names:
            self.lane_max_counts[lane] = max(self.lane_max_counts[lane], lane_counts[lane])
            if lane_counts[lane] > 0:
                self.lane_active_frames[lane] += 1

        self.frame_trend.append((frame_index, vehicle_count))

        overall_density = classify_density(vehicle_count)
        lane_densities = {lane: classify_density(cnt) for lane, cnt in lane_counts.items()}

        # Running (cumulative-so-far) signal + score per lane: each
        # lane's numbers (max/avg/utilization seen SO FAR) feed its own
        # Traffic Score, and that lane's score alone decides its own
        # RED/GREEN signal. No lane is compared against the other -
        # this is why the two traffic lights behave as fully separate
        # systems. Updating every frame lets the output video and the
        # live camera feed both show it evolving in real time.
        lane_signals = {}
        lane_scores = {}
        for lane in self.lane_names:
            signal_info, score_info = build_lane_result(
                lane,
                max_vehicles=self.lane_max_counts[lane],
                avg_vehicles=self.lane_totals[lane] / self.frame_count,
                utilization=self.lane_active_frames[lane] / self.frame_count,
            )
            lane_signals[lane] = signal_info
            lane_scores[lane] = score_info

        return {
            "vehicle_count": vehicle_count,
            "lane_counts": lane_counts,
            "overall_density": overall_density,
            "lane_densities": lane_densities,
            "lane_signals": lane_signals,
            "lane_scores": lane_scores,
        }

    def summary(self):
        avg_vehicles = (
            round(self.sum_vehicles_per_frame / self.frame_count, 2)
            if self.frame_count else 0
        )
        overall_density_label = classify_density(round(avg_vehicles))

        # Down-sample the trend so the embedded chart JSON stays small
        # even for long videos.
        trend = self.frame_trend
        if len(trend) > config.MAX_CHART_POINTS:
            step = len(trend) / config.MAX_CHART_POINTS
            sampled_idx = [int(i * step) for i in range(config.MAX_CHART_POINTS)]
            trend = [trend[i] for i in sampled_idx]

        road_utilization_pct = (
            round((self.sum_vehicles_per_frame / (self.frame_count * max(self.max_vehicles_in_frame, 1))) * 100, 1)
            if self.frame_count else 0
        )

        # --------------------------------------------------------
        # FINAL PER-LANE TRAFFIC SIGNAL
        # --------------------------------------------------------
        # Each lane's final Traffic Score (from its own whole-video
        # max/average/utilization/density) decides its own final
        # RED/GREEN signal via the >=50% rule:
        #   Score >= 50%  -> GREEN
        #   Score <  50%  -> RED
        # Every lane is computed independently - lanes are never
        # compared against each other, and there is no shared/winner
        # signal state. Lane 1 and Lane 2 are two fully separate
        # signal systems.
        final_signals = {}
        lane_scores = {}
        for lane in self.lane_names:
            signal_info, score_info = build_lane_result(
                lane,
                max_vehicles=self.lane_max_counts[lane],
                avg_vehicles=(self.lane_totals[lane] / self.frame_count) if self.frame_count else 0,
                utilization=(self.lane_active_frames[lane] / self.frame_count) if self.frame_count else 0,
            )
            final_signals[lane] = signal_info
            lane_scores[lane] = score_info

        return {
            "total_frames": self.frame_count,
            "total_vehicle_detections": self.total_detections,
            "max_vehicles_in_frame": self.max_vehicles_in_frame,
            "average_vehicles_per_frame": avg_vehicles,
            "overall_density": overall_density_label,
            "class_totals": self.class_totals,
            "lane_totals": self.lane_totals,
            "lane_names": self.lane_names,
            "frame_trend": trend,
            "road_utilization_pct": road_utilization_pct,
            "final_signals": final_signals,
            "lane_scores": lane_scores,
        }