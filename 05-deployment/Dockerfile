FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install uv and then dependencies
RUN pip install uv && uv pip install --system --no-cache-dir -r requirements.txt

COPY ["hw-Q4.py", "pipeline_v1.bin", "./"]

# Expose port
EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "hw-Q4:app"]
