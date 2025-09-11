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
import pytesseract




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
        self.license_plate_model = YOLO(license_plate_model_path)  # NEW
        self.vehicle_timestamps = defaultdict(list)  # Keep track of timestamps for each tracked vehicle
        


    def _initialize_classifiers(self):
        if self.color_classifier is None:
            self.color_classifier = ColorClassifier()
        if self.model_classifier is None:
            self.model_classifier = ModelClassifier()

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

    def _read_license_plate(self, lp_image):
        gray = cv2.cvtColor(lp_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(thresh, config="--psm 7")
        text = text.strip()
        return text if text else None

    def process_frame_base64(self, frame_base64, frame_timestamp, run_inference=True):
        frame = self._decode_image_base64(frame_base64)
        if frame is not None:
            return self.process_frame(frame, frame_timestamp, run_inference=run_inference)
        else:
            return {"error": "Failed to decode the base64 image"}

    def process_frame(self, frame, frame_timestamp, run_inference=True):
        """
        If run_inference is False, we skip YOLO/ByteTrack and just redraw the last detections.
        """
        self._initialize_classifiers()

        response = {
            "number_of_vehicles_detected": 0,
            "detected_vehicles": []
        }


        # INFERENCE FRAME: full YOLO + tracker
        bright = self._increase_brightness(frame)
        results = self.model.track(bright, persist=True, tracker="bytetrack.yaml")

        annotated_frame = frame.copy()
        annotator = Annotator(annotated_frame, line_width=2, font_size=14, pil=False)

        if results and results[0] and results[0].boxes is not None:
            r0 = results[0]
            names = r0.names

            # (a) For box labels we want xyxy
            xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().tolist()
            clss  = r0.boxes.cls.cpu().tolist()
            
            xywh      = r0.boxes.xywh.cpu().numpy()
            track_ids = r0.boxes.id
            track_ids = track_ids.int().cpu().tolist() if track_ids is not None else [None] * len(clss)
  
            for box, cls, conf, tid in zip(xyxy, clss, confs, track_ids):
                cls_name = names[int(cls)]
                id_text = f"ID:{tid}" if tid is not None else "ID:-"
                label = f"{cls_name} {conf:.2f} {id_text}"
                annotator.box_label(box, label, color=colors(int(cls), True))


            # Finalize Annotator -> NumPy (RGB) then convert to BGR for OpenCV ops
            annotated_frame = annotator.result()
            # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            # 3) Polylines + crops/classification from ORIGINAL frame
            H, W = frame.shape[:2]
            
            for (cx, cy, w, h), cls, conf, tid in zip(xywh, clss, confs, track_ids):
                cls_name = names[int(cls)]
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
                
                lp_results = self.license_plate_model(vehicle_frame)[0]
                license_plate_data = []
                for lp_box in lp_results.boxes.data.tolist():
                    vx1, vy1, vx2, vy2, score, class_id = lp_box
                    # Convert coords to original frame
                    full_x1 = int(x1 + vx1)
                    full_y1 = int(y1 + vy1)
                    full_x2 = int(x1 + vx2)
                    full_y2 = int(y1 + vy2)
                    
                    # Crop the plate region from the original frame (optional)
                    lp_crop = frame[full_y1:full_y2, full_x1:full_x2]
                    lp_text = self._read_license_plate(lp_crop)
                    lp_crop_b64 = self._encode_image_base64(lp_crop)

                    license_plate_data.append({
                        "bbox": [full_x1, full_y1, full_x2, full_y2],  # full-frame coordinates
                        "text": lp_text,
                        "confidence": score,
                        "crop_base64": lp_crop_b64
                    })
                                
                vehicle_frame_base64 = self._encode_image_base64(vehicle_frame)
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
                    "vehicle_frame_base64": vehicle_frame_base64,
                    "color_info": json.dumps(color_info),
                    "model_info": json.dumps(model_info)
                }
                
                matched_plate = None
                for lp in license_plate_data:
                    lp_x1, lp_y1, lp_x2, lp_y2 = lp["bbox"]

                    # Check if license plate bbox is inside vehicle bbox
                    if (lp_x1 >= x1 and lp_y1 >= y1 and lp_x2 <= x2 and lp_y2 <= y2):
                        matched_plate = lp
                        break  # stop at first match (you could also keep the highest confidence)

                if matched_plate:
                    # Crop plate region from original frame
                    lp_crop = frame[int(matched_plate["bbox"][1]):int(matched_plate["bbox"][3]),
                                    int(matched_plate["bbox"][0]):int(matched_plate["bbox"][2])]
                    lp_crop_b64 = self._encode_image_base64(lp_crop)

                    # Add license plate info to this vehicle
                    item["license_plate"] = {
                        "text": matched_plate["text"],
                        "confidence": float(matched_plate["confidence"]),
                        "bbox": matched_plate["bbox"],
                        "crop_base64": lp_crop_b64
                    }
                response["detected_vehicles"].append(item)

                
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
