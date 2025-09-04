import json
import math
import cv2
import base64
from collections import defaultdict
from ultralytics import YOLO
import numpy as np
from ultralytics.utils.plotting import colors, Annotator
from VehicleDetectionTracker.color_classifier.classifier import Classifier as ColorClassifier
from VehicleDetectionTracker.model_classifier.classifier import Classifier as ModelClassifier
from datetime import datetime, timedelta

class VehicleDetectionTracker:

    def __init__(self, model_path="yolov11n.pt"):
        """
        Initialize the VehicleDetection class.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        # Load the YOLO model and set up data structures for tracking.
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])  # History of vehicle tracking
        self.detected_vehicles = set()  # Set of detected vehicles
        self.color_classifier = None
        self.model_classifier = None
        self.vehicle_timestamps = defaultdict(list)  # Keep track of timestamps for each tracked vehicle

        # NEW: caches so we can annotate skipped frames without re-running the model
        self._last_dets = []          # list of dicts: {cx, cy, w, h, cls, conf, tid}
        self._names_cache = None      # class names from last inference


    def _initialize_classifiers(self):
        if self.color_classifier is None:
            self.color_classifier = ColorClassifier()
        if self.model_classifier is None:
            self.model_classifier = ModelClassifier()

    def _map_direction_to_label(self, direction):
        direction_ranges = {
            (-math.pi / 8, math.pi / 8): "Right",
            (math.pi / 8, 3 * math.pi / 8): "Bottom Right",
            (3 * math.pi / 8, 5 * math.pi / 8): "Bottom",
            (5 * math.pi / 8, 7 * math.pi / 8): "Bottom Left",
            (7 * math.pi / 8, -7 * math.pi / 8): "Left",
            (-7 * math.pi / 8, -5 * math.pi / 8): "Top Left",
            (-5 * math.pi / 8, -3 * math.pi / 8): "Top",
            (-3 * math.pi / 8, -math.pi / 8): "Top Right"
        }
        for angle_range, label in direction_ranges.items():
            if angle_range[0] <= direction <= angle_range[1]:
                return label
        return "Unknown"

    def _encode_image_base64(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode()
        return image_base64
    
    def _decode_image_base64(self, image_base64):
        try:
            image_data = base64.b64decode(image_base64)
            image_np = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_np, flags=cv2.IMREAD_COLOR)
            return image
        except Exception:
            return None
        
    def _increase_brightness(self, image, factor=1.5):
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)

    def _convert_meters_per_second_to_kmph(self, meters_per_second):
        return meters_per_second * 3.6

    def process_frame_base64(self, frame_base64, frame_timestamp, run_inference=True):
        frame = self._decode_image_base64(frame_base64)
        if frame is not None:
            return self.process_frame(frame, frame_timestamp, run_inference=run_inference)
        else:
            return {"error": "Failed to decode the base64 image"}
            
    def _annotate_from_cached_dets(self, frame):
        """
        Draw boxes/labels on the given frame using self._last_dets and self._names_cache.
        No tracker update; just a quick overlay for skipped frames.
        """
        annotated_frame = frame.copy()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        annotator = Annotator(annotated_frame, line_width=2, font_size=14, pil=False)
        if self._last_dets and self._names_cache:
            H, W = frame.shape[:2]
            for det in self._last_dets:
                cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]
                cls, conf, tid = det["cls"], det["conf"], det["tid"]

                # clamp and draw
                x1 = int(max(0, cx - w/2)); y1 = int(max(0, cy - h/2))
                x2 = int(min(W, cx + w/2)); y2 = int(min(H, cy + h/2))
                if x2 <= x1 or y2 <= y1:
                    continue
                id_text = f"ID:{tid}" if tid is not None else "ID:-"
                label = f"{self._names_cache[int(cls)]} {conf:.2f} {id_text}"
                annotator.box_label([x1, y1, x2, y2], label, color=colors(int(cls), True))

        annotated_frame = annotator.result()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        return annotated_frame

    def process_frame(self, frame, frame_timestamp, run_inference=True):
        """
        If run_inference is False, we skip YOLO/ByteTrack and just redraw the last detections.
        """
        self._initialize_classifiers()

        response = {
            "number_of_vehicles_detected": 0,
            "detected_vehicles": []
        }

        # SKIPPED FRAME: reuse last detections for fast overlay
        if not run_inference:
            annotated_frame = self._annotate_from_cached_dets(frame)
            response["annotated_frame"] = annotated_frame
            response["original_frame"] = frame

            # We also echo back the last known detections (no new classifications on skipped frames)
            for det in self._last_dets:
                response["number_of_vehicles_detected"] += 1
                response["detected_vehicles"].append({
                    "vehicle_id": det["tid"],
                    "vehicle_type": str(self._names_cache[int(det["cls"])]) if self._names_cache else "unknown",
                    "detection_confidence": float(det["conf"]),
                    "vehicle_coordinates": {
                        "x": float(det["cx"]), "y": float(det["cy"]),
                        "width": float(det["w"]), "height": float(det["h"])
                    },
                    # no fresh crops/classification on skipped frames, keep from last or empty
                    "color_info": det.get("color_info", "{}"),
                    "model_info": det.get("model_info", "{}")
                })
            return response

        # INFERENCE FRAME: full YOLO + tracker
        bright = self._increase_brightness(frame)
        results = self.model.track(bright, persist=True, tracker="bytetrack.yaml")

        annotated_frame = frame.copy()
        annotator = Annotator(annotated_frame, line_width=2, font_size=14, pil=False)

        # reset cache for this frame
        self._last_dets = []
        self._names_cache = None

        if results and results[0] and results[0].boxes is not None:
            r0 = results[0]
            names = r0.names
            self._names_cache = names

            # (a) For box labels we want xyxy
            xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().tolist()
            clss  = r0.boxes.cls.cpu().tolist()
            
            xywh      = r0.boxes.xywh.cpu().numpy()
            track_ids = r0.boxes.id
            track_ids = track_ids.int().cpu().tolist() if track_ids is not None else [None] * len(clss)

            # Draw labeled boxes on the original frame
            allowed_classes = {"bus", "truck", "car", "bicycle", "motorcycle"}

            for box, cls, conf, tid in zip(xyxy, clss, confs, track_ids):
                cls_name = names[int(cls)]
                if cls_name not in allowed_classes:
                    continue
                id_text = f"ID:{tid}" if tid is not None else "ID:-"
                label = f"{cls_name} {conf:.2f} {id_text}"
                annotator.box_label(box, label, color=colors(int(cls), True))


            # Finalize Annotator -> NumPy (RGB) then convert to BGR for OpenCV ops
            annotated_frame = annotator.result()
            # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            # 3) Polylines + crops/classification from ORIGINAL frame
            H, W = frame.shape[:2]

            allowed_classes = {"bus", "truck", "car", "bicycle", "motorcycle"}
            
            for (cx, cy, w, h), cls, conf, tid in zip(xywh, clss, confs, track_ids):
                cls_name = names[int(cls)]
                if cls_name not in allowed_classes:
                    continue
                id_text = f"ID:{tid}" if tid is not None else "ID:-"
                label = f"{cls_name} {conf:.2f} {id_text}"
                annotator.box_label(box, label, color=colors(int(cls), True))
                # Update per-track history and draw trajectory if we have an ID
                if tid is not None:
                    trk = self.track_history[tid]
                    trk.append((float(cx), float(cy)))
                    if len(trk) > 30:
                        trk.pop(0)
                    pts = np.hstack(trk).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], isClosed=False,
                                color=colors(int(cls), True), thickness=2)

                # Safe crop from ORIGINAL frame (not the brightened one)
                x1 = int(cx - w / 2); y1 = int(cy - h / 2)
                x2 = int(cx + w / 2); y2 = int(cy + h / 2)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate box after clamping

                vehicle_frame = frame[y1:y2, x1:x2]
                if vehicle_frame.size == 0:
                    continue

                # Classifiers
                color_info = self.color_classifier.predict(vehicle_frame)
                model_info = self.model_classifier.predict(vehicle_frame)

                # Build response item
                response["number_of_vehicles_detected"] += 1
                item = {
                    "vehicle_id": tid,  # may be None on first few frames before tracker locks
                    "vehicle_type": str(names[int(cls)]),
                    "detection_confidence": float(conf),
                    "vehicle_coordinates": {
                        "x": float(cx), "y": float(cy),
                        "width": float(w), "height": float(h)
                    },
                    "color_info": json.dumps(color_info),
                    "model_info": json.dumps(model_info)
                }
                response["detected_vehicles"].append(item)

                # Cache minimal det info for skipped frames
                self._last_dets.append({
                    "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h),
                    "cls": int(cls), "conf": float(conf), "tid": tid,
                    "color_info": json.dumps(color_info),
                    "model_info": json.dumps(model_info)
                })

        response["annotated_frame"] = annotated_frame
        response["original_frame"]  = frame
        return response

    
    def process_video(self, video_path, result_callback=None,
                      output_path="output_annotated.mp4",
                      show_preview=False,
                      resize_to=None,  
                      scale=None):      
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        writer = None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        if fps <= 0: fps = 25

        import time
        t0 = time.perf_counter(); frames = 0
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if resize_to is not None:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            elif scale is not None:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                
            video_time_seconds = frame_idx / fps

            # NEW: run model every 2 frames (even frames)
            run_inference = (frame_idx % 3 == 0)

            response = self.process_frame(frame, datetime.now(), run_inference=run_inference)
            response["frame_index"] = frame_idx
            response["video_time_seconds"] = video_time_seconds

            annotated = response.get("annotated_frame")
            if annotated is not None:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    if not writer.isOpened():
                        cap.release()
                        raise RuntimeError(f"Could not open writer for: {output_path}")
                writer.write(annotated)

                if show_preview:
                    cv2.imshow("YOLOv8 + ByteTrack (stride=2)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if result_callback is not None:
                result_callback(response)

            # simple FPS meter
            frames += 1
            print('---------frame-------')
            print(frames)
            if frames % 60 == 0:
                dt = time.perf_counter() - t0
                print(f"~FPS: {frames/dt:.1f}")
                t0 = time.perf_counter(); frames = 0
            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
