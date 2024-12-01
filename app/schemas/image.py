from pydantic import BaseModel

class ImageData(BaseModel):
    image_b64: str  # Base64 encoded image string
