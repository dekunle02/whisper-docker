FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    git \
    ffmpeg \
    python3 \
    python3-pip \
    pkg-config \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# clone whisper.cpp
# RUN git clone https://github.com/ggerganov/whisper.cpp.git && \
#     cd whisper.cpp && \
#     ./models/download-ggml-model.sh base.en && \
#     make && \
#     cp libwhisper.so /usr/local/lib/ && \
#     cd .. && \
#     rm -rf whisper.cpp

# Set library path
# ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH


RUN git clone  https://github.com/ggerganov/whisper.cpp.git && \
    cd whisper.cpp && \
    # Build the static library
    make libwhisper.a && \
    # Create directory structure expected by pywhispercpp
    mkdir -p /usr/local/include/whisper && \
    cp *.h /usr/local/include/whisper/ && \
    cp libwhisper.a /usr/local/lib/ && \
    cd ..
    # rm -rf whisper.cpp

# Set required environment variables for pywhispercpp
ENV WHISPER_CPP_LIB=/usr/local/lib/libwhisper.a
ENV WHISPER_CPP_INCLUDE=/usr/local/include



COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 8000


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]