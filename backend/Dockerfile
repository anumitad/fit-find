# Use a lightweight Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository "deb http://deb.debian.org/debian bullseye-backports main" && \
    apt-get update && apt-get install -y ffmpeg


RUN apt-get update && apt-get install -y ffmpeg



# Copy the rest of the backend code
COPY . .

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
