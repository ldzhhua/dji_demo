"""Simple FastAPI backend for serving change detection results."""

from fastapi import FastAPI
from pathlib import Path
from typing import Optional

from inference.run_inference import infer

app = FastAPI(title="Change Detection API")


@app.post("/detect")
def detect(input_path: str, model_path: Optional[str] = None):
    """Run change detection on ``input_path`` using the given model."""
    input_file = Path(input_path)
    if model_path is not None:
        model = Path(model_path)
    else:
        model = Path("model.pth")

    output_dir = Path("predictions")
    try:
        infer(model, input_file.parent, output_dir)
    except Exception as exc:
        return {"error": str(exc)}

    return {
        "message": "Inference finished",
        "input": str(input_file),
        "model": str(model),
        "results": str(output_dir),
    }
