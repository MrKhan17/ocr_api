# import torch
# import numpy as np
# from PIL import Image

# # Load your custom YOLOv8 model (assumed to be saved as a .pt file)
# yolo_model = torch.load("path_to_your_custom_yolov8_model.pt")
# yolo_model.eval()  # Set the model to evaluation mode

# def get_bounding_boxes(image: Image) -> list:
#     """Detect objects using custom YOLO model and return bounding boxes."""
#     img_np = np.array(image)  # Convert PIL Image to numpy array
#     img_tensor = torch.from_numpy(img_np).float()  # Convert to tensor (make sure to follow YOLOv8 input format)
    
#     # Make predictions with the custom YOLOv8 model
#     with torch.no_grad():  # Disable gradient calculation during inference
#         results = yolo_model([img_tensor])
    
#     # Extract bounding boxes
#     boxes = results.xywh[0][:, :4].cpu().numpy()  # Assuming `xywh` gives [x, y, w, h] for each bounding box
#     return boxes
