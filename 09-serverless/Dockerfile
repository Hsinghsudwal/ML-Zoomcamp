FROM agrigorev/model-2025-hairstyle:v1

RUN python3.13 -m pip install --no-cache-dir numpy pillow onnxruntime

COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]

