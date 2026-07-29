FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY demo.py .

# Pre-warm model cache at build time so first run is instant
RUN python demo.py --fast --no-pause || true

CMD ["python", "demo.py"]
