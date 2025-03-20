import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
import io
import torch.nn.functional as F
from vit_model import VIT_B16_224
import cv2
import numpy as np
import dlib
import os

# Global variables for reuse
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Diseased"]
FACE_DETECTOR = None
LANDMARK_PREDICTOR = None


def initialize_detectors():
    global FACE_DETECTOR, LANDMARK_PREDICTOR
    # Initialize the face detector and landmark predictor
    FACE_DETECTOR = dlib.get_frontal_face_detector()
    LANDMARK_PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("Face and eye detectors initialized")


def load_model(model_path):
    model = VIT_B16_224(num_classes=2)

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=DEVICE)

    # Remove the 'module.' prefix from state_dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v

    # Load the modified state dict
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def create_side_by_side_eyes(left_eye_img, right_eye_img):
    """
    Create a side-by-side image of both eyes and resize to 224x224

    Args:
        left_eye_img: PIL Image of left eye
        right_eye_img: PIL Image of right eye

    Returns:
        PIL Image with both eyes side by side, resized to 224x224
    """
    # Create a new image with both eyes side by side
    combined_width = left_eye_img.width + right_eye_img.width
    max_height = max(left_eye_img.height, right_eye_img.height)
    combined_img = Image.new('RGB', (combined_width, max_height))

    # Paste the eyes side by side
    combined_img.paste(left_eye_img, (0, 0))
    combined_img.paste(right_eye_img, (left_eye_img.width, 0))

    # Resize to 224x224 for the model
    combined_img_resized = combined_img.resize((224, 224), Image.LANCZOS)

    return combined_img_resized


def detect_and_crop_eyes(image, return_visualization=False):
    """
    Detect and crop left and right eyes from an image

    Args:
        image: PIL Image or numpy array
        return_visualization: Whether to return visualization image

    Returns:
        tuple: (left_eye_image, right_eye_image, visualization_image, side_by_side_eyes, ear_values) as PIL Images
               or (None, None, None, None, None) if detection fails
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = FACE_DETECTOR(gray)

    if len(faces) == 0:
        print("No face detected")
        return None, None, None, None, None

    # Get the first face
    face = faces[0]

    # Get facial landmarks
    landmarks = LANDMARK_PREDICTOR(gray, face)

    # Eye landmarks indices
    # Left eye: points 36-41
    # Right eye: points 42-47
    left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

    # Calculate EAR values
    left_ear = calculate_ear(left_eye_points)
    right_ear = calculate_ear(right_eye_points)
    ear_values = {"left": left_ear, "right": right_ear}

    # Get bounding boxes with padding
    def get_eye_bbox(points, padding=10):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        min_x, max_x = min(x_coords) - padding, max(x_coords) + padding
        min_y, max_y = min(y_coords) - padding, max(y_coords) + padding
        return (max(0, min_x), max(0, min_y), max_x, max_y)

    left_eye_bbox = get_eye_bbox(left_eye_points)
    right_eye_bbox = get_eye_bbox(right_eye_points)

    # Crop eyes
    left_eye_img = img_np[left_eye_bbox[1]:left_eye_bbox[3], left_eye_bbox[0]:left_eye_bbox[2]]
    right_eye_img = img_np[right_eye_bbox[1]:right_eye_bbox[3], right_eye_bbox[0]:right_eye_bbox[2]]

    # Check if the cropped images are valid (not empty)
    if left_eye_img.size == 0 or right_eye_img.size == 0:
        print("Invalid eye crop dimensions")
        return None, None, None, None, None

    # Convert to PIL Images
    left_eye_pil = Image.fromarray(left_eye_img)
    right_eye_pil = Image.fromarray(right_eye_img)

    # Create side-by-side eyes image resized to 224x224
    side_by_side_eyes = create_side_by_side_eyes(left_eye_pil, right_eye_pil)

    # Create visualization if requested
    vis_img = None
    if return_visualization:
        vis_img = visualize_landmarks(img_np, landmarks, face)
        vis_img_pil = Image.fromarray(vis_img)
        return left_eye_pil, right_eye_pil, vis_img_pil, side_by_side_eyes, ear_values

    return left_eye_pil, right_eye_pil, None, side_by_side_eyes, ear_values


def visualize_landmarks(image, landmarks, face=None):
    """
    Draw facial landmarks on an image for visualization

    Args:
        image: numpy array image
        landmarks: dlib facial landmarks
        face: dlib face rectangle (optional)

    Returns:
        numpy array image with landmarks visualized
    """
    vis_img = image.copy()

    # Draw face rectangle if provided
    if face:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw all facial landmarks
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1)

    # Highlight eye landmarks
    for i in range(36, 48):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(vis_img, (x, y), 3, (0, 0, 255), -1)

    # Draw eye contours
    left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], np.int32)
    right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], np.int32)

    cv2.polylines(vis_img, [left_eye_pts], True, (255, 255, 0), 1)
    cv2.polylines(vis_img, [right_eye_pts], True, (255, 255, 0), 1)

    return vis_img


def calculate_ear(eye_points):
    """
    Calculate the eye aspect ratio for the given eye points

    Args:
        eye_points: list of (x,y) coordinates for an eye

    Returns:
        float: eye aspect ratio
    """
    # Compute vertical distances
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))

    # Compute horizontal distance
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))

    # Calculate EAR
    ear = (A + B) / (2.0 * C)
    return ear


def preprocess_image_from_path(image_path, return_visualization=False):
    try:
        # Open the image file
        img = Image.open(image_path).convert('RGB')

        # Detect and crop eyes
        left_eye, right_eye, vis_img, side_by_side_eyes, ear_values = detect_and_crop_eyes(img, return_visualization)

        if left_eye is None or right_eye is None:
            # If eye detection fails, raise an exception
            raise ValueError("Eyes not detected")

        # Process both eyes separately
        left_eye_tensor = preprocess_image(left_eye)
        right_eye_tensor = preprocess_image(right_eye)

        # Also preprocess the side-by-side image for the model
        side_by_side_tensor = preprocess_image(side_by_side_eyes)

        # Return both eye tensors and additional data
        result = {
            "tensors": [left_eye_tensor, right_eye_tensor, side_by_side_tensor],
            "ear_values": ear_values,
            "side_by_side_eyes": side_by_side_eyes
        }

        if return_visualization:
            result["visualization"] = vis_img

        return result
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def preprocess_image_from_bytes(image_bytes, return_visualization=False):
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Detect and crop eyes
        left_eye, right_eye, vis_img, side_by_side_eyes, ear_values = detect_and_crop_eyes(img, return_visualization)

        if left_eye is None or right_eye is None:
            # If eye detection fails, raise an exception
            raise ValueError("Eyes not detected")

        # Process both eyes separately
        left_eye_tensor = preprocess_image(left_eye)
        right_eye_tensor = preprocess_image(right_eye)

        # Also preprocess the side-by-side image for the model
        side_by_side_tensor = preprocess_image(side_by_side_eyes)

        # Return both eye tensors and additional data
        result = {
            "tensors": [left_eye_tensor, right_eye_tensor, side_by_side_tensor],
            "ear_values": ear_values,
            "side_by_side_eyes": side_by_side_eyes
        }

        if return_visualization:
            result["visualization"] = vis_img

        return result
    except Exception as e:
        print(f"Error processing image: {e}")
        # Re-raise the exception to be caught by the calling function
        raise


def preprocess_image(img):
    # Define the preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply preprocessing
    img_tensor = preprocess(img)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def get_probabilities(img_tensor):
    global MODEL
    # If img_tensor is a list of tensors (left eye, right eye, side-by-side)
    if isinstance(img_tensor, list):
        all_probs = []
        for tensor in img_tensor:
            # Move image to device
            tensor = tensor.to(DEVICE)

            # Get prediction
            with torch.no_grad():
                outputs = MODEL(tensor)
                # Apply softmax to convert to probabilities
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())

        # Average the probabilities from all images
        all_probs = torch.cat(all_probs, dim=0)
        avg_probs = torch.mean(all_probs, dim=0).unsqueeze(0)
        return avg_probs
    # If img_tensor is a dict with tensors
    elif isinstance(img_tensor, dict) and "tensors" in img_tensor:
        return get_probabilities(img_tensor["tensors"])
    else:
        # Original single image processing
        # Move image to device
        img_tensor = img_tensor.to(DEVICE)

        # Get prediction
        with torch.no_grad():
            outputs = MODEL(img_tensor)
            # Apply softmax to convert to probabilities
            probs = F.softmax(outputs, dim=1)

        return probs.cpu()


def initialize_model(model_path):
    global MODEL
    MODEL = load_model(model_path)
    initialize_detectors()
    print(f"Model loaded on {DEVICE}")


def process_images(image_paths, save_visualization=True):
    # Create output directory if it doesn't exist
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # Process each image and collect probabilities
    all_probs = []
    for i, image_path in enumerate(image_paths):
        print(f"Processing image: {image_path}")
        img_tensor = preprocess_image_from_path(image_path, return_visualization=True)

        if img_tensor is None:
            print(f"Skipping {image_path} due to processing error")
            continue

        # Save the side-by-side eyes image if requested
        if save_visualization and "side_by_side_eyes" in img_tensor:
            # Get base filename without path and extension
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = os.path.join(output_dir, f"{base_filename}_eyes_224x224.jpg")
            img_tensor["side_by_side_eyes"].save(output_filename)
            print(f"Saved side-by-side eyes image to {output_filename}")

            # Also save the visualization with landmarks if available
            if "visualization" in img_tensor:
                vis_filename = os.path.join(output_dir, f"{base_filename}_landmarks.jpg")
                img_tensor["visualization"].save(vis_filename)
                print(f"Saved landmarks visualization to {vis_filename}")

        # Get probabilities
        probs = get_probabilities(img_tensor)
        all_probs.append(probs)

    if not all_probs:
        print("No images were successfully processed.")
        return None

    # Stack all probability tensors and calculate the mean
    all_probs = torch.cat(all_probs, dim=0)
    avg_probs = torch.mean(all_probs, dim=0)

    # Get the predicted class from averaged probabilities
    _, predicted = torch.max(avg_probs, 0)
    predicted_class_idx = predicted.item()
    predicted_class = CLASS_NAMES[predicted_class_idx]

    return {
        "prediction": predicted_class,
        "class_index": predicted_class_idx,
        "confidence": float(avg_probs[predicted_class_idx].item() * 100),
        "probabilities": avg_probs.numpy().tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Detect strabismus from eye images")
    parser.add_argument("--images", required=True, nargs='+', help="Paths to input images")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--save-visualization", action="store_true", help="Save visualization images")
    args = parser.parse_args()

    # Initialize the model
    initialize_model(args.model)

    # Process the images
    result = process_images(args.images, save_visualization=args.save_visualization)

    if result:
        # Print results
        print("\nResults:")
        print(f"Average probabilities: {result['probabilities']}")
        print(f"Final prediction: {result['prediction']} (Class {result['class_index']})")
        print(f"Confidence: {result['confidence']:.2f}%")


if __name__ == "__main__":
    main()
