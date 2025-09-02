import json
import math
import cv2
import base64
from collections import defaultdict
from ultralytics import YOLO
import numpy as np
from ultralytics.utils.plotting import colors
from VehicleDetectionTracker.color_classifier.classifier import Classifier as ColorClassifier
from VehicleDetectionTracker.model_classifier.classifier import Classifier as ModelClassifier
from datetime import datetime

class VehicleDetectionTracker:

    def __init__(self, model_path="yolov8n.pt"):
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


    def _initialize_classifiers(self):
        if self.color_classifier is None:
            self.color_classifier = ColorClassifier()
        if self.model_classifier is None:
            self.model_classifier = ModelClassifier()

    def _map_direction_to_label(self, direction):
        # Define direction ranges in radians and their corresponding labels
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
        return "Unknown"  # Return "Unknown" if the direction doesn't match any defined range


    def _encode_image_base64(self, image):
        """
        Encode an image as base64.

        Args:
            image (numpy.ndarray): The image to be encoded.

        Returns:
            str: Base64-encoded image.
        """
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode()
        return image_base64
    
    def _decode_image_base64(self, image_base64):
        """
        Decode a base64-encoded image.

        Args:
            image_base64 (str): Base64-encoded image data.

        Returns:
            numpy.ndarray or None: Decoded image as a numpy array or None if decoding fails.
        """
        try:
            image_data = base64.b64decode(image_base64)
            image_np = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_np, flags=cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            return None
        
    def _increase_brightness(self, image, factor=1.5):
        """
        Increases the brightness of an image by multiplying its pixels by a factor.

        :param image: The input image in numpy array format.
        :param factor: The brightness increase factor. A value greater than 1 will increase brightness.
        :return: The image with increased brightness.
        """
        brightened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return brightened_image

    def _convert_meters_per_second_to_kmph(self, meters_per_second):
        # 1 m/s is approximately 3.6 km/h
        kmph = meters_per_second * 3.6
        return kmph

    def process_frame_base64(self, frame_base64, frame_timestamp):
        """
        Process a base64-encoded frame to detect and track vehicles.

        Args:
            frame_base64 (str): Base64-encoded input frame for processing.

        Returns:
            dict or None: Processed information including tracked vehicles' details and the annotated frame in base64,
            or an error message if decoding fails.
        """
        frame = self._decode_image_base64(frame_base64)
        if frame is not None:
            return self.process_frame(frame, frame_timestamp)
        else:
            return {
                "error": "Failed to decode the base64 image"
            }
            
    def process_video(self, video_path, result_callback, output_path="output_annotated.mp4"):
        cap = cv2.VideoCapture(video_path)

        writer = None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            timestamp = datetime.now()
            response = self.process_frame(frame, timestamp)

            annotated_frame = response.get("annotated_frame", None)
            if annotated_frame is not None:
                if writer is None:
                    h, w = annotated_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                writer.write(annotated_frame)

            result_callback(response)

        cap.release()
        if writer is not None:
            writer.release()


    def process_frame(self, frame, frame_timestamp):
        self._initialize_classifiers()
        response = {
            "number_of_vehicles_detected": 0,
            "detected_vehicles": []
        }

        # Default annotated frame is a copy of the input (in case there are no detections)
        annotated_frame = frame.copy()

        results = self.model.track(self._increase_brightness(frame), persist=True, tracker="bytetrack.yaml")
        if results is not None and results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            conf_list = results[0].boxes.conf.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            names = results[0].names

            # Use YOLO's plotted image as the base for our annotations
            annotated_frame = results[0].plot()

            for box, track_id, cls, conf in zip(boxes, track_ids, clss, conf_list):
                x, y, w, h = box
                label = str(names[cls])
                bbox_color = colors(cls, True)
                track_thickness = 2

                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=bbox_color, thickness=track_thickness)

                if track_id not in self.vehicle_timestamps:
                    self.vehicle_timestamps[track_id] = {"timestamps": [], "positions": []}

                self.vehicle_timestamps[track_id]["timestamps"].append(frame_timestamp)
                self.vehicle_timestamps[track_id]["positions"].append((x, y))

                timestamps = self.vehicle_timestamps[track_id]["timestamps"]
                positions = self.vehicle_timestamps[track_id]["positions"]
                speed_kph, reliability, direction_label, direction = None, 0.0, None, None

                if len(timestamps) >= 2:
                    delta_t_list, distance_list = [], []
                    for i in range(1, len(timestamps)):
                        t1, t2 = timestamps[i - 1], timestamps[i]
                        delta_t = t2.timestamp() - t1.timestamp()
                        if delta_t > 0:
                            x1, y1 = positions[i - 1]
                            x2, y2 = positions[i]
                            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                            delta_t_list.append(delta_t)
                            distance_list.append(distance)

                    speeds = [d / dt for d, dt in zip(distance_list, delta_t_list) if dt > 0]
                    avg_speed_mps = sum(speeds) / len(speeds) if speeds else None
                    speed_kph = self._convert_meters_per_second_to_kmph(avg_speed_mps) if avg_speed_mps is not None else None

                    initial_x, initial_y = positions[0]
                    final_x, final_y = positions[-1]
                    direction = math.atan2(final_y - initial_y, final_x - initial_x)
                    direction_label = self._map_direction_to_label(direction)

                    reliability = 0.5 if len(timestamps) < 5 else (0.7 if len(timestamps) < 10 else 1.0)

                # Safe crop for vehicle frame (avoid empty slices on borders)
                H, W = frame.shape[:2]
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                vehicle_frame = frame[y1:y2, x1:x2]
                if vehicle_frame.size == 0:
                    continue

                color_info = self.color_classifier.predict(vehicle_frame)
                model_info = self.model_classifier.predict(vehicle_frame)

                self.detected_vehicles.add(track_id)
                response["number_of_vehicles_detected"] += 1
                response["detected_vehicles"].append({
                    "vehicle_id": track_id,
                    "vehicle_type": label,
                    "detection_confidence": float(conf.item()),
                    "vehicle_coordinates": {"x": float(x.item()), "y": float(y.item()), "width": float(w.item()), "height": float(h.item())},
                    "color_info": json.dumps(color_info),
                    "model_info": json.dumps(model_info),
                    "speed_info": {
                        "kph": speed_kph,
                        "reliability": reliability,
                        "direction_label": direction_label,
                        "direction": direction
                    }
                })

        response["annotated_frame"] = annotated_frame  
        response["original_frame"] = frame             

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

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if resize_to is not None:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            elif scale is not None:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            response = self.process_frame(frame, datetime.now())

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
                    cv2.imshow("YOLOv8 + ByteTrack", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if result_callback is not None:
                result_callback(response)

            # simple FPS meter
            frames += 1
            if frames % 60 == 0:
                dt = time.perf_counter() - t0
                print(f"~FPS: {frames/dt:.1f}")
                t0 = time.perf_counter(); frames = 0

        cap.release()
        if writer is not None:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
