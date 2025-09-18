import cv2
import numpy as np

# Define basic color ranges in HSV (can refine these)
COLOR_RANGES = {
    "red":      [(0, 50, 50), (10, 255, 255)],
    "orange":   [(11, 50, 50), (25, 255, 255)],
    "yellow":   [(26, 50, 50), (35, 255, 255)],
    "green":    [(36, 50, 50), (85, 255, 255)],
    "cyan":     [(86, 50, 50), (95, 255, 255)],
    "blue":     [(96, 50, 50), (130, 255, 255)],
    "purple":   [(131, 50, 50), (160, 255, 255)],
    "pink":     [(161, 50, 50), (170, 255, 255)],
    "white":    [(0, 0, 200), (180, 30, 255)],
    "gray":     [(0, 0, 50), (180, 30, 200)],
    "black":    [(0, 0, 0), (180, 255, 50)],
    "silver":   [(0, 0, 120), (180, 30, 220)]
}

def closest_color(hsv_val, color_ranges):
    """Map HSV value to the closest named color."""
    h, s, v = hsv_val
    best_name, best_dist = "unknown", 999
    for name, (lower, upper) in color_ranges.items():
        hl, sl, vl = lower
        hu, su, vu = upper
        center = np.array([(hl + hu) // 2, (sl + su) // 2, (vl + vu) // 2])
        dist = np.linalg.norm(np.array([h, s, v]) - center)
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name

def get_car_color(crop: np.ndarray) -> str:
    """
    Estimate the main color of a cropped car image.
    Uses k-means for dominant color and maps to nearest named color.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    # Focus on lower 2/3
    hsv = hsv[int(h/3):, :]

    # Blur slightly to reduce noise
    hsv = cv2.GaussianBlur(hsv, (5,5), 0)

    # Flatten pixels
    pixels = hsv.reshape(-1, 3)

    # Remove very dark or low-saturation pixels (windows, shadows, tires)
    mask = (pixels[:,2] > 50) & (pixels[:,1] > 30)
    filtered = pixels[mask]
    if len(filtered) == 0:
        return "unknown"

    # K-means clustering to find dominant color
    pixels_float = np.float32(filtered)
    k = 2
    _, labels, centers = cv2.kmeans(
        pixels_float, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Dominant cluster
    dominant_hsv = centers[np.argmax(np.bincount(labels.flatten()))].astype(int)

    # Map to closest named color
    color_name = closest_color(dominant_hsv, COLOR_RANGES)
    return color_name

def main():
    car_dir = 'car-images/'
    image_paths = [
        'blue-bmw.jpg',
        'black-merc.jpg',
        'purple-merc.jpg',  
        'red-merc.jpg',
        'silver-merc.jpg'
    ]

    for filename in image_paths:
        expected_color = filename.split('-')[0]
        full_path = f'{car_dir}{filename}'
        image = cv2.imread(full_path)
        if image is None:
            print(f"Failed to load {filename}")
            continue

        predicted_color = get_car_color(image)
        if predicted_color == expected_color:
            print(f'{filename} was predicted correctly')
        else:
            print(f'{filename} was predicted incorrectly')
            print(f'Predicted Color: {predicted_color}, Actual Color: {expected_color}')

if __name__ == "__main__":
    main()
