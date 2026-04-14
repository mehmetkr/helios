FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir httpx
COPY load_gen.py .

CMD ["python", "load_gen.py"]
