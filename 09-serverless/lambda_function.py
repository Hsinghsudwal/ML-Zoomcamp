import onnxruntime as ort
import numpy as np
from PIL import Image
from io import BytesIO
import urllib.request

TARGET_SIZE = (200, 200)

def download_image(url):
    with urllib.request.urlopen(url) as resp:
        img_data = resp.read()
    img = Image.open(BytesIO(img_data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def preprocess(img):
    img = img.resize(TARGET_SIZE, Image.NEAREST)
    x = np.array(img).astype("float32")

    # x = x / 255 - 0.5  → range [-0.5, 0.5]
    x = x / 255.0 - 0.5

    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0).astype("float32")
    return x

# Load ONNX
session = ort.InferenceSession("hair_classifier_empty.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def lambda_handler(event, context=None):
    url = event["url"]
    img = download_image(url)
    x = preprocess(img)

    pred = session.run([output_name], {input_name: x})[0][0][0]
    return float(pred)
