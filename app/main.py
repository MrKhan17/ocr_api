from fastapi import FastAPI
from app.schemas.image import ImageData
# from app.models.detection import get_bounding_boxes
from app.models.ocr import ocr_on_bboxes
from app.utils.decode import decode_image
from app.utils.image_processing import resize_and_pad_image_opencv
from fastapi.exceptions import HTTPException
import base64
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = FastAPI()

# Load YOLOv8 model
model = YOLO("app/weights/materials_seg_best.pt") 


# Define paths
model_path = 'app/weights/pytorch_model.bin'
config_path = 'app/weights/config.json'  # Provide the config file path if available

# Load the processor (tokenizer + feature extractor)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")

# Load the model
model_ocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
model_ocr.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model_ocr.eval()  # Set the model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ocr.to(device)


@app.post("/detect_and_ocr/")
async def detect_and_ocr(data: ImageData):
    try:
        # Convert base64 image to PIL Image
        image_data = base64.b64decode(data.image_b64)
        # Convert bytes to a NumPy array
        nparr = np.frombuffer(image_data, np.uint8)

        # Decode the array into an image (OpenCV format)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # image = Image.open(BytesIO(image_data))
        results = model(image, task='detect',save=True) 
        # Iterate through detections and crop each object
        crops = []
        for detection in results[0].boxes.xyxy:  # If using ultralytics, boxes.xyxy contains bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, detection)  # Convert coordinates to integers
            crop = image[y_min:y_max, x_min:x_max]  # Crop the region from the original image
            crops.append(crop)
            crop = resize_and_pad_image_opencv(crop)

            pixel_values = processor(images=crop, return_tensors="pt").pixel_values
            with torch.no_grad():
                output_ids = model_ocr.generate(pixel_values)

            # Decode the output to text
            predicted_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            print(predicted_text)
            # print(f"Image Path: {row['image']}, Text: {row['text']}", "Predicted Text:", predicted_text)
        # Step 1: Detect bounding boxes
        # bounding_boxes = get_bounding_boxes(image)
        
        # if len(bounding_boxes) == 0:
        #     raise HTTPException(status_code=400, detail="No objects detected in the image.")
        
        # # Step 2: OCR processing on bounding boxes
        # ocr_results = ocr_on_bboxes(image, bounding_boxes)
        
        return {"bounding_boxes": 'bounding_boxes', "ocr_results": 'ocr_results'}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))