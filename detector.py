"""
detector.py
Loads the provided ONNX model once and runs YOLOv8-style detection on
individual frames. Handles preprocessing (letterbox resize), inference,
and postprocessing (confidence filtering + NMS + box rescaling) based
on the model's ACTUAL input/output shapes and embedded class names -
nothing here is hardcoded to a specific YOLO version or class set.
"""
import ast
import cv2
import numpy as np
import onnxruntime as ort

import config


class ModelLoadError(Exception):
    """Raised when the ONNX model cannot be loaded."""


class VehicleDetector:
    def __init__(self, model_path=None):
        model_path = str(model_path or config.MODEL_PATH)

        try:
            available = ort.get_available_providers()
            providers = [p for p in config.EXECUTION_PROVIDERS_PRIORITY if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]

            self.session = ort.InferenceSession(model_path, providers=providers)
            self.active_provider = self.session.get_providers()[0]
            print(f"[Detector] Execution provider in use: {self.active_provider}")
        except Exception as exc:
            raise ModelLoadError(f"Failed to load ONNX model at '{model_path}': {exc}") from exc

        # Inspect the real input shape rather than assuming 640x640.
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        shape = input_info.shape
        try:
            self.input_h = int(shape[2]) if isinstance(shape[2], int) else config.INPUT_SIZE
            self.input_w = int(shape[3]) if isinstance(shape[3], int) else config.INPUT_SIZE
        except (IndexError, TypeError, ValueError):
            self.input_h = self.input_w = config.INPUT_SIZE

        self.output_names = [o.name for o in self.session.get_outputs()]

        # Pull class names from the model's own metadata if present,
        # otherwise fall back to the config default.
        self.class_names = self._load_class_names()
        self.num_classes = len(self.class_names)

    def _load_class_names(self):
        try:
            meta = self.session.get_modelmeta()
            names_str = meta.custom_metadata_map.get("names")
            if names_str:
                parsed = ast.literal_eval(names_str)
                return {int(k): str(v) for k, v in parsed.items()}
        except Exception as exc:
            print(f"[Detector] Could not parse embedded class names, using fallback: {exc}")
        return dict(config.FALLBACK_CLASS_NAMES)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _letterbox(self, frame):
        """Resize frame to the model's input size while preserving aspect
        ratio, padding with grey. Returns the padded image plus the scale
        and padding offsets needed to map boxes back to the original frame.
        """
        h, w = frame.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = self.input_w - new_w
        pad_h = self.input_h - new_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, scale, left, top

    def _preprocess(self, frame):
        padded, scale, pad_x, pad_y = self._letterbox(frame)
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))          # HWC -> CHW
        img = np.expand_dims(img, axis=0)            # add batch dim
        return np.ascontiguousarray(img), scale, pad_x, pad_y

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------
    def _postprocess(self, output, scale, pad_x, pad_y, frame_w, frame_h,
                      conf_threshold, iou_threshold):
        """Handles the standard Ultralytics YOLOv8 export layout:
        output shape (1, 4 + num_classes, num_anchors), boxes in
        cx,cy,w,h (letterboxed-image pixel space), class scores already
        in [0,1] (no separate objectness column).
        """
        preds = output[0]                 # (channels, num_anchors)
        preds = preds.transpose(1, 0)     # (num_anchors, channels)

        box_xywh = preds[:, :4]
        class_scores = preds[:, 4:4 + self.num_classes]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        keep = confidences >= conf_threshold
        box_xywh = box_xywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        if len(box_xywh) == 0:
            return []

        # cx,cy,w,h (letterboxed space) -> x1,y1,x2,y2 (letterboxed space)
        cx, cy, w, h = box_xywh[:, 0], box_xywh[:, 1], box_xywh[:, 2], box_xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        boxes_xyxy = np.stack([x1, y1, w, h], axis=1)  # x,y,w,h for cv2.dnn.NMSBoxes

        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold,
        )

        detections = []
        if len(indices) == 0:
            return detections

        indices = np.array(indices).flatten()
        for i in indices:
            bx, by, bw, bh = boxes_xyxy[i]
            # Undo letterbox padding + scaling to map back to original frame
            ox1 = (bx - pad_x) / scale
            oy1 = (by - pad_y) / scale
            ox2 = (bx + bw - pad_x) / scale
            oy2 = (by + bh - pad_y) / scale

            ox1 = max(0, min(frame_w - 1, ox1))
            oy1 = max(0, min(frame_h - 1, oy1))
            ox2 = max(0, min(frame_w - 1, ox2))
            oy2 = max(0, min(frame_h - 1, oy2))

            cls_id = int(class_ids[i])
            detections.append({
                "bbox": (int(ox1), int(oy1), int(ox2), int(oy2)),
                "class_id": cls_id,
                "class_name": self.class_names.get(cls_id, f"class_{cls_id}"),
                "confidence": float(confidences[i]),
            })
        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame, conf_threshold=None, iou_threshold=None):
        conf_threshold = config.CONFIDENCE_THRESHOLD if conf_threshold is None else conf_threshold
        iou_threshold = config.NMS_IOU_THRESHOLD if iou_threshold is None else iou_threshold

        frame_h, frame_w = frame.shape[:2]
        tensor, scale, pad_x, pad_y = self._preprocess(frame)

        try:
            outputs = self.session.run(self.output_names, {self.input_name: tensor})
        except Exception as exc:
            raise RuntimeError(f"ONNX inference failed: {exc}") from exc

        return self._postprocess(
            outputs[0], scale, pad_x, pad_y, frame_w, frame_h,
            conf_threshold, iou_threshold
        )
