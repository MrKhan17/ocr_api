import base64
from io import BytesIO
from PIL import Image

def decode_image(b64_string: str) -> Image:
    """Decode base64 string to a PIL Image."""
    img_data = base64.b64decode(b64_string)
    img = Image.open(BytesIO(img_data))
    return img
