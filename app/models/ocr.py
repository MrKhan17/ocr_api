from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Initialize TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

def ocr_on_bboxes(image: Image, boxes: list) -> list:
    """Process bounding boxes using OCR."""
    text_results = []
    for (xmin, ymin, xmax, ymax) in boxes:
        # Crop image using the bounding box
        cropped_img = image.crop((xmin, ymin, xmax, ymax))
        
        # Preprocess for OCR
        pixel_values = processor(cropped_img, return_tensors="pt").pixel_values
        output_ids = ocr_model.generate(pixel_values)
        
        # Decode OCR output
        decoded_text = processor.decode(output_ids[0], skip_special_tokens=True)
        text_results.append(decoded_text)
    return text_results
