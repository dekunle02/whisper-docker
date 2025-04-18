import os
import tempfile
import shutil
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydub import AudioSegment
from whisper_cpp_python import Whisper

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR / "models"
MODEL_NAME = "ggml-small.en.bin"
MODEL_PATH = MODEL_DIR / MODEL_NAME


# --- Whisper Model Loading ---
whisper_model = None
if MODEL_PATH.exists():
    try:
        whisper_model = Whisper(
            # model_path=str(MODEL_PATH), n_threads=max(4, os.cpu_count() // 2)
            model_path=str(MODEL_PATH),
            n_threads=2,
        )

        print(f"Successfully loaded Whisper model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        # prevent the app from starting if the model fails to load
        # raise RuntimeError(f"Could not load Whisper model: {e}") from e
else:
    print(
        f"Warning: Whisper model not found at {MODEL_PATH}. Transcription endpoint will fail."
    )


# --- FastAPI Application ---
app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Whisper.cpp Live"}


@app.post("/transcribe/")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Receives an audio file, converts it to 16-bit WAV,
    transcribes it using whisper.cpp, and returns the text.
    """
    print(f"Received file: size: {audio_file.size / (1024 * 1024):.2f} MB")
    # --- Start Timing ---
    start_time = time.perf_counter()

    if whisper_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Whisper model not loaded. Check server logs. Model expected at {MODEL_PATH}",
        )

    # Create temporary directories for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_audio_path = temp_dir_path / (audio_file.filename or "input_audio")
        output_wav_path = temp_dir_path / "output.wav"

        try:
            with open(input_audio_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)
            print(f"Saved uploaded file to: {input_audio_path}")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving uploaded file: {e}",
            ) from e
        finally:
            await audio_file.close()

        # Convert audio file to 16-bit WAV using pydub
        try:
            print(f"Attempting to load audio from: {input_audio_path}")
            audio = AudioSegment.from_file(input_audio_path)
            print(
                f"Original audio - Channels: {audio.channels}, Sample width: {audio.sample_width}, Frame rate: {audio.frame_rate}"
            )
            audio = audio.set_frame_rate(16000)
            audio = audio.set_channels(1)
            audio.export(output_wav_path, format="wav")

            if not output_wav_path.exists():
                raise RuntimeError("WAV file was not created.")

        except Exception as e:
            print(f"Error during audio conversion: {e}")
            # Try listing files for debugging in container
            try:
                print(f"Files in {temp_dir_path}: {list(temp_dir_path.iterdir())}")
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error converting audio file. Ensure it's a valid format (e.g., MP3, WAV, M4A, OGG). Details: {e}",
            ) from e

        # Transcribe using whisper.cpp
        try:
            result = whisper_model.transcribe(str(output_wav_path), language="en")

            # Check if result is structured differently (depends on whisper_cpp_python version)
            transcription_text = ""
            if isinstance(result, dict) and "text" in result:
                transcription_text = result["text"].strip()
            elif isinstance(result, str):  # Older versions might return just the string
                transcription_text = result.strip()
            else:
                # If the structure is different, log it and adapt
                print(f"Unexpected transcription result format: {result}")
                transcription_text = str(result).strip()  # Fallback

            # --- End Timing ---
            end_time = time.perf_counter()
            transcription_duration = end_time - start_time
            print(
                f"Transcription successfully took: {transcription_duration:.4f} seconds"
            )

            return {"transcription": transcription_text}

        except Exception as e:
            print(f"Error during transcription: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during transcription: {e}",
            ) from e


# Optional: Add logic to run Uvicorn directly if the script is executed
# Useful for local testing without Docker
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
