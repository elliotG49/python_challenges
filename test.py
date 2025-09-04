# --- NEW: add these imports at top of your calling script ---
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from datetime import datetime
import cv2, os, json

# --- set up Mongo ---
client = MongoClient("mongodb://localhost:27017")
db = client["trafficDB"]
col_videos = db.videos
col_frames = db.frames
col_detections = db.detections
col_tracks = db.tracks

video_path = r"C:\Users\ellio\Downloads\13958312-hd_1920_1080_25fps.mp4"
output_path = r"c:\Users\ellio\Videos\output_annotated.avi"

# --- probe video to get metadata for the videos doc ---
cap_probe = cv2.VideoCapture(video_path)
if not cap_probe.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")
fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25
ret, first_frame = cap_probe.read()
cap_probe.release()
if not ret:
    raise RuntimeError("Could not read first frame to get dimensions")
h, w = first_frame.shape[:2]

# --- create videos doc before processing ---
video_doc = {
    "path": os.fspath(video_path),
    "output_path": os.fspath(output_path),
    "width": int(w),
    "height": int(h),
    "fps": float(fps),
    "model": "yolo11n.pt",
    "tracker": "bytetrack",
    "allowed_classes": ["bus","truck","car","bicycle","motorcycle"],
    "started_at": datetime.utcnow(),
}
video_id = col_videos.insert_one(video_doc).inserted_id

# --- helper: finalize video at end ---
def _finish_video_doc():
    col_videos.update_one({"_id": video_id}, {"$set": {"finished_at": datetime.utcnow()}})
    # compute avg_conf from sum_conf / detections_count (if we used incremental tallies)
    col_tracks.update_many(
        {"video_id": video_id, "detections_count": {"$gt": 0}},
        [{"$set": {"avg_conf": {"$divide": ["$sum_conf", "$detections_count"]}}}]
    )

# --- your detector (unchanged) ---
vehicle_detection = VehicleDetectionTracker(model_path="yolo11n.pt")

# --- result callback writes to Mongo ---
def result_callback(result):
    """
    result = {
      "frame_index": int,
      "video_time_seconds": float,
      "number_of_vehicles_detected": int,
      "detected_vehicles": [{...}]
    }
    """
    fi = result.get("frame_index", 0)

    # Only write frames where inference actually ran in your code:
    # run_inference = (frame_idx % 3 == 0)
    if fi % 3 != 0:
        return

    # 1) frames (upsert)
    frame_doc = {
        "video_id": video_id,
        "frame_index": fi,
        "video_time_s": float(result.get("video_time_seconds", 0.0)),
        "num_detections": int(result.get("number_of_vehicles_detected", 0)),
    }
    col_frames.update_one(
        {"video_id": video_id, "frame_index": fi},
        {"$set": frame_doc},
        upsert=True
    )

    # 2) detections (insert_many)
    det_docs = []
    track_updates = []
    for v in result.get("detected_vehicles", []):
        tid = v.get("vehicle_id")
        coords = v.get("vehicle_coordinates", {}) or {}
        # detection doc
        det_docs.append({
            "video_id": video_id,
            "frame_index": fi,
            "track_id": tid,
            "class": v.get("vehicle_type"),
            "conf": float(v.get("detection_confidence", 0.0)),
            "bbox_xywh": {
                "x": float(coords.get("x", 0.0)),
                "y": float(coords.get("y", 0.0)),
                "width": float(coords.get("width", 0.0)),
                "height": float(coords.get("height", 0.0)),
            },
            "centroid": {"x": float(coords.get("x", 0.0)), "y": float(coords.get("y", 0.0))},
            "color_info": (json.loads(v.get("color_info") or "{}")),
            "model_info": (json.loads(v.get("model_info") or "{}")),
        })

        # live track roll-up (skip None track ids)
        if tid is not None:
            color_label = (json.loads(v.get("color_info") or "{}")).get("label")
            model_label = (json.loads(v.get("model_info") or "{}")).get("label")
            inc = {
                "detections_count": 1,
                "sum_conf": float(v.get("detection_confidence", 0.0))
            }
            if color_label:
                inc[f"color_votes.{color_label}"] = 1
            if model_label:
                inc[f"model_votes.{model_label}"] = 1

            track_updates.append(UpdateOne(
                {"video_id": video_id, "track_id": tid},
                {
                    "$setOnInsert": {
                        "video_id": video_id,
                        "track_id": tid,
                        "class": v.get("vehicle_type"),
                        "first_frame": fi
                    },
                    "$max": {"last_frame": fi},
                    "$inc": inc
                },
                upsert=True
            ))

    if det_docs:
        col_detections.insert_many(det_docs, ordered=False)
    if track_updates:
        col_tracks.bulk_write(track_updates, ordered=False)

# --- run processing ---
try:
    vehicle_detection.process_video(
        video_path,
        result_callback=result_callback,
        output_path=output_path,
        show_preview=True
    )
finally:
    _finish_video_doc()

print(f"Saved annotated video to: {output_path}\nvideo_id = {video_id}")
