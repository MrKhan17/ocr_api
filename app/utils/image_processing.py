import cv2
import numpy as np

def resize_and_pad_image_opencv(img: np.ndarray, new_width: int = 384) -> np.ndarray:
    # Get original dimensions
    original_height, original_width = img.shape[:2]
    
    # Calculate new dimensions maintaining aspect ratio
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)
    
    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Create a new square canvas (new_width x new_width) with a white background
    padded_img = np.full((new_width, new_width, 3), 255, dtype=np.uint8)  # White background
    
    # Calculate the vertical offset to center the resized image
    top = (new_width - new_height) // 2
    
    # Paste the resized image onto the center of the canvas
    padded_img[top:top+new_height, 0:new_width] = resized_img

    return padded_img
