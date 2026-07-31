FROM python:3.10-slim

WORKDIR /app

# Copy Requirements and Install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy All Source Code
COPY . .

# Expose Render Port (Render overrides this with environment PORT, but standardizing 8000 is good)
EXPOSE 8000

# Run FastAPI via Uvicorn
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
