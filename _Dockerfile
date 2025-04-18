# Use an official Python runtime matching the suspected version and newer distro
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    wget \
    ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Check CMake version (for debugging)
RUN cmake --version

# Set the working directory in the container
WORKDIR /app

# Upgrade pip, setuptools, and wheel first
RUN pip install --upgrade pip setuptools wheel

# Copy the requirements file into the container
COPY requirements.txt .

# --- Attempt to fix CMake build ---
# Set CMAKE_ARGS to try and force the policy version suggested by the error
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# Install Python dependencies from requirements.txt
# This will attempt to build whisper-cpp-python using the CMAKE_ARGS above
RUN pip install --no-cache-dir -r requirements.txt

# --- Download Whisper Model ---
# Create a directory for models
RUN mkdir models
# Download the desired model (e.g., base.en). Find URLs on Hugging Face (ggerganov/whisper.cpp)
ARG MODEL_URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
ARG MODEL_DEST=/app/models/ggml-base.en.bin
RUN wget --progress=bar:force -O ${MODEL_DEST} ${MODEL_URL}

# Unset CMAKE_ARGS if not needed later (optional, good practice)
ENV CMAKE_ARGS=""

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]