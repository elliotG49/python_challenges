from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker

video_path = r"C:\Users\ellio\Downloads\4608275-uhd_3840_2160_24fps.mp4"
output_path = r"c:\Users\ellio\Videos\output_annotated.mp4"  # change as you like

vehicle_detection = VehicleDetectionTracker(model_path="yolov8n.pt")

# Optional: keep or simplify to avoid huge console logs
def result_callback(result):
    # Print per-frame header with frame index and video time
    print({
        "frame_index": result.get("frame_index"),
        "video_time_seconds": round(result.get("video_time_seconds", 0.0), 3),
        "number_of_vehicles_detected": result["number_of_vehicles_detected"],
    })

    # Optionally also print them per-vehicle
    print([
        {
            "frame_index": result.get("frame_index"),
            "video_time_seconds": round(result.get("video_time_seconds", 0.0), 3),
            "vehicle_id": vehicle["vehicle_id"],
            "vehicle_type": vehicle["vehicle_type"],
            "detection_confidence": vehicle["detection_confidence"],
            "color_info": vehicle["color_info"],
            "model_info": vehicle["model_info"],
        }
        for vehicle in result.get("detected_vehicles", [])
    ])


# Save the annotated video (no preview window)
vehicle_detection.process_video(video_path, result_callback, output_path, show_preview=True, resize_to=(1920, 1080))

print(f"Saved annotated video to: {output_path}")
