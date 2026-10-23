
import os
import tempfile
from pathlib import Path
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from speechbrain.inference.enhancement import SpectralMaskEnhancement

app = FastAPI(title="MetricGAN+ Speech Enhancement")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = BASE_DIR / "pretrained_models" / "metricgan-plus"

print("Loading MetricGAN+ model...")

model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir=str(MODEL_DIR)
)

print("MetricGAN+ model loaded successfully!")


@app.get("/health")
def health():
    return {"status": "running", "model": "MetricGAN+"}


@app.post("/enhance")
async def enhance_audio(audio: UploadFile = File(...)):
    input_path = None
    output_path = None

    try:
        if not audio.filename:
            raise HTTPException(
                status_code=400,
                detail="Please select an audio file."
            )

        data = await audio.read()

        if not data:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty."
            )

        # Create a temporary input file and close it before loading.
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp_input:
            temp_input.write(data)
            input_path = str(
                Path(temp_input.name).resolve()
            )

        # Use forward slashes for Windows path compatibility.
        input_path = input_path.replace("\\", "/")

        print("Processing uploaded audio...")

        with torch.inference_mode():
            enhanced = model.enhance_file(input_path)

        if enhanced.ndim == 1:
            enhanced = enhanced.unsqueeze(0)

        # Create the output file.
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp_output:
            output_path = str(
                Path(temp_output.name).resolve()
            )

        output_path = output_path.replace("\\", "/")

        sf.write(
            output_path,
            enhanced.detach().cpu().squeeze().numpy(),
            16000,
            format="WAV"
)

        print("Audio enhancement completed successfully!")

        # Remove the original temporary input.
        if os.path.exists(input_path):
            os.remove(input_path)
            input_path = None

        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename="enhanced_speech.wav",
            background=BackgroundTask(
                os.remove, output_path
            )
        )

    except HTTPException:
        raise

    except Exception as exc:
        print(f"Audio enhancement error: {exc}")

        raise HTTPException(
            status_code=500,
            detail=f"Enhancement failed: {exc}"
        )

    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)


# Serve the frontend after defining the API routes.
if STATIC_DIR.exists():
    app.mount(
        "/",
        StaticFiles(
            directory=str(STATIC_DIR),
            html=True
        ),
        name="frontend"
    )
