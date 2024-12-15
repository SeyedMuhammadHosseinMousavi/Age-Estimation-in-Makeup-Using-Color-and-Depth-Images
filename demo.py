import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_ubyte

# Step 1: Load RGB and Depth Images
def load_images(rgb_path, depth_path):
    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if rgb_image is None or depth_image is None:
        raise FileNotFoundError("Error: Cannot load images. Check paths.")
    return rgb_image, depth_image

# Step 2: Face Detection and Cropping
def detect_and_crop_face(rgb_image, depth_image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_rgb, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        raise ValueError("Error: No face detected in RGB image.")
    
    x, y, w, h = faces[0]  # Assume one face
    face_rgb = gray_rgb[y:y+h, x:x+w]
    face_depth = depth_image[y:y+h, x:x+w]
    return face_rgb, face_depth

# Step 3: Feature Extraction
def extract_features(face_rgb, face_depth):
    # Canny Edge Detection for RGB Image
    edges_rgb = cv2.Canny(face_rgb, 50, 150)
    # Entropy Filter for Depth Image
    filtered_depth = entropy(img_as_ubyte(face_depth), disk(5))
    return edges_rgb, filtered_depth

# Step 4: Age Estimation with Adjusted Scaling
def estimate_age(edges_rgb, filtered_depth, base_age=25, adjustment_factor=10):
    sum_rgb = np.sum(edges_rgb)  # Sum of edge pixels
    sum_depth = np.sum(filtered_depth)  # Sum of depth features
    total_sum = sum_rgb + sum_depth  # Combined features
    
    # Debugging Info
    print(f"Sum of Edges (RGB): {sum_rgb}")
    print(f"Sum of Depth Features: {sum_depth}")

    # Dynamic Scaling Around Base Age
    normalized_score = (total_sum % adjustment_factor) / adjustment_factor
    estimated_age = base_age + (normalized_score * adjustment_factor - adjustment_factor / 2)
    
    return round(estimated_age)

# Main Pipeline Execution
if __name__ == "__main__":
    try:
        # Step 1: Load Images
        rgb_image, depth_image = load_images('c1.jpg', 'd11.jpg')  # Replace file paths
        
        # Step 2: Face Detection and Cropping
        face_rgb, face_depth = detect_and_crop_face(rgb_image, depth_image)
        
        # Step 3: Feature Extraction
        edges_rgb, filtered_depth = extract_features(face_rgb, face_depth)
        
        # Step 4: Estimate Age
        estimated_age = estimate_age(edges_rgb, filtered_depth)
        
        # Output Estimated Age
        print(f"Estimated Age: {estimated_age} years")
    
    except Exception as e:
        print(f"Error: {e}")
