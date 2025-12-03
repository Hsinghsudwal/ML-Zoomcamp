# src/inference.py
import numpy as np
import tensorflow as tf
import os

class TFLiteModel:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model.tflite")
            model_path = os.path.abspath(model_path)

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, raw):
        """
        Example normalization/reshape stub.
        raw: list or numpy array shaped to model input (e.g. [H,W,C] or [1,H,W,C])
        Return: numpy array ready for interpreter
        """
        arr = np.array(raw, dtype=np.float32)
        # If model expects batch dim, ensure present
        if len(arr.shape) == len(self.input_details[0]['shape']) - 1:
            arr = np.expand_dims(arr, 0)
        # Ensure dtype matches
        arr = arr.astype(self.input_details[0]['dtype'])
        return arr

    def predict(self, raw_input):
        inp = self.preprocess(raw_input)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        return out.tolist()


# src/lambda_handler.py
from inference import TFLiteModel
import os
import json

# Load model once (cold-start warm cache)
MODEL_PATH = os.environ.get("TFLITE_MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "model", "model.tflite"))
model = TFLiteModel(model_path=MODEL_PATH)

def handler(event, context=None):
    """
    Lambda-style handler that accepts an `event` dict.
    Example event:
      { "input": [[...]] }  OR  { "input": <json-serializable single item> }
    Returns a dict: {"statusCode": int, "body": {...}}
    """
    try:
        # If event is the raw HTTP body string (FastAPI forwarding), try parse
        if isinstance(event, str):
            try:
                event = json.loads(event)
            except Exception:
                event = {"input": event}

        input_data = event.get("input")
        if input_data is None:
            return {"statusCode": 400, "body": {"error": "missing 'input' key"}}

        prediction = model.predict(input_data)
        return {"statusCode": 200, "body": {"prediction": prediction}}
    except Exception as e:
        return {"statusCode": 500, "body": {"error": str(e)}}


# src/serve.py
from fastapi import FastAPI, Request
from lambda_function import handler
import uvicorn
import json

app = FastAPI(title="TFLite Serverless Simulator")

@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()
    # forward to the lambda-style handler
    result = handler(payload)
    # If body is already object, return it as JSON
    return result

if __name__ == "__main__":
    uvicorn.run("src.serve:app", host="0.0.0.0", port=8080, reload=False)


# src/utils.py
import numpy as np
def normalize_image(img):
    # example stub; adapt to how your model was trained
    arr = np.array(img).astype('float32') / 255.0
    return arr


# Dockerfile -- multi-stage for smaller final image
FROM python:3.10-slim as build

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + model
FROM python:3.10-slim
WORKDIR /app

# Copy installed packages from build stage
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# App files
COPY src/ src/
COPY model/ model/

ENV PYTHONPATH=/app/src
ENV TFLITE_MODEL_PATH=/app/model/model.tflite

EXPOSE 8080

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8080"]

version: "3.8"
services:
  tflite-server:
    build: .
    image: ml-serverless:tflite-local
    container_name: tflite-server
    ports:
      - "8080:8080"
    environment:
      - TFLITE_MODEL_PATH=/app/model/model.tflite
    volumes:
      - ./model:/app/model:ro
      - ./src:/app/src:ro

  # Optional LocalStack for Lambda emulation (local-only)
  localstack:
    image: localstack/localstack:2.1.1
    container_name: localstack
    environment:
      - SERVICES=lambda,apigateway
      - DEBUG=1
      - DOCKER_HOST=unix:///var/run/docker.sock
    ports:
      - "4566:4566"
      - "4571:4571"
    volumes:
      - "/var/run/docker.sock:/var/run/docker.sock"
      - "./localstack:/tmp/localstack"


# build_deployment.sh
zip -r deployment.zip src model requirements.txt
# create-function (example)
awslocal lambda create-function \
  --function-name tflite-local \
  --handler lambda_handler.handler \
  --runtime python3.10 \
  --zip-file fileb://deployment.zip \
  --role arn:aws:iam::000000000000:role/lambda-role
# invoke
awslocal lambda invoke --function-name tflite-local out.json --payload '{"input":[[0.1,0.2,0.3]]}'
cat out.json





