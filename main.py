import io
from typing import List
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from pywhispercpp.model import Model
import numpy as np
import subprocess
import uvicorn
import av

app = FastAPI()
# MODEL_PATH = "/app/whisper.cpp/models/ggml-base.en.bin"
model = Model("base.en", n_threads=6)


class TranscriptionResponse(BaseModel):
    text: str
    segments: List[dict]


def convert_audio_to_wav(audio_data: bytes, target_sr: int = 16000) -> np.ndarray:
    """
    Convert audio bytes to 16kHz WAV using ffmpeg in-memory
    """
    try:
        # Create input container
        input_container = av.open(io.BytesIO(audio_data))
        input_stream = input_container.streams.audio[0]

        # Create output container in memory
        output_buffer = io.BytesIO()
        output_container = av.open(output_buffer, mode="w", format="wav")

        # Add audio stream
        output_stream = output_container.add_stream("pcm_s16le", rate=target_sr)

        # Initialize resampler
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=target_sr,
        )

        # Read all audio frames, resample, and write
        for frame in input_container.decode(input_stream):
            frame.pts = None
            for resampled_frame in resampler.resample(frame):
                output_container.mux(resampled_frame)

        # Close the output container
        output_container.close()

        # Get the bytes from the buffer
        wav_bytes = output_buffer.getvalue()

        # Convert to numpy array
        audio_np = np.frombuffer(wav_bytes[44:], dtype=np.int16)  # Skip WAV header
        return audio_np.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Audio conversion failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Initialize Whisper model on startup"""
    global whisper_model
    try:
        # Initialize whisper-cpp model
        # You can change the model size as needed (tiny, base, small, medium, large)
        whisper_model = whisper_cpp.Whisper(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load Whisper model: {str(e)}")
        raise


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile):
    """
    Endpoint to transcribe audio files
    Accepts: mp3, wav, m4a, webm
    Returns: Transcription text and segments
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read file content
    content = await file.read()

    try:
        # Convert audio to proper format
        audio_array = convert_audio_to_wav(content)

        # Perform transcription
        # result = whisper_model.transcribe(audio_array)

        segments = model.transcribe(audio_array, speed_up=True)
        all_text = ""
        for segment in segments:
            all_text += " " + segment.text

        # Format response
        # response = TranscriptionResponse(
        #     text=result["text"],
        #     segments=[
        #         {
        #             "text": segment["text"],
        #             "start": segment["start"],
        #             "end": segment["end"],
        #             "confidence": segment.get("confidence", 0.0),
        #         }
        #         for segment in result["segments"]
        #     ],
        # )
        file.close()
        return segment

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
