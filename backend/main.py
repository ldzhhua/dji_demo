"""FastAPI application for the UAV change detection platform."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dataset.create_dataset import create_dataset
from inference.run_inference import infer
from training.train_model import train

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

TaskStatus = Literal["queued", "running", "completed", "failed"]
TaskKind = Literal["dataset", "training", "inference"]


@dataclass
class Job:
    id: str
    kind: TaskKind
    title: str
    status: TaskStatus = "queued"
    progress: int = 0
    message: str = "Waiting to start"
    created_at: str = field(default_factory=lambda: now_iso())
    updated_at: str = field(default_factory=lambda: now_iso())
    result: dict[str, Any] | None = None


class DatasetRequest(BaseModel):
    input_dir: str = Field(..., examples=["data/raw"])
    output_dir: str = Field(..., examples=["data/processed"])
    val_ratio: float = Field(0.2, ge=0, le=1)
    seed: int | None = Field(None, examples=[42])


class TrainingRequest(BaseModel):
    config_path: str = Field(..., examples=["configs/changeformer.py"])
    work_dir: str = Field(..., examples=["runs/changeformer"])


class InferenceRequest(BaseModel):
    model_path: str = Field("model.pth", examples=["runs/best.pth"])
    input_dir: str = Field(..., examples=["data/inference"])
    output_dir: str = Field("predictions", examples=["predictions/latest"])


app = FastAPI(
    title="UAV Change Detection Platform",
    description="Dataset preparation, OpenCD training and inference dashboard.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, Job] = {}
jobs_lock = Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_job(job: Job) -> dict[str, Any]:
    return asdict(job)


def save_job(job: Job) -> None:
    job.updated_at = now_iso()
    with jobs_lock:
        jobs[job.id] = job


def create_job(kind: TaskKind, title: str) -> Job:
    job = Job(id=uuid4().hex, kind=kind, title=title)
    save_job(job)
    return job


def run_job(job_id: str, action: Callable[[Job], dict[str, Any]]) -> None:
    with jobs_lock:
        job = jobs[job_id]
    try:
        job.status = "running"
        job.progress = 20
        job.message = "Running"
        save_job(job)
        job.result = action(job)
        job.status = "completed"
        job.progress = 100
        job.message = "Completed successfully"
    except Exception as exc:  # noqa: BLE001 - API should return captured job failures.
        job.status = "failed"
        job.progress = 100
        job.message = str(exc)
    save_job(job)


@app.get("/")
def index():
    """Serve the dashboard shell."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": app.title,
        "opencd_available": is_opencd_available(),
        "jobs": len(jobs),
    }


@app.get("/api/overview")
def overview():
    with jobs_lock:
        job_list = list(jobs.values())
    completed = sum(1 for job in job_list if job.status == "completed")
    failed = sum(1 for job in job_list if job.status == "failed")
    running = sum(1 for job in job_list if job.status == "running")
    return {
        "metrics": {
            "total_jobs": len(job_list),
            "completed_jobs": completed,
            "failed_jobs": failed,
            "running_jobs": running,
        },
        "capabilities": [
            "Dataset train/validation split",
            "OpenCD training launcher",
            "OpenCD inference launcher",
            "Modern operations dashboard",
        ],
    }


@app.get("/api/jobs")
def list_jobs():
    with jobs_lock:
        sorted_jobs = sorted(
            jobs.values(), key=lambda item: item.created_at, reverse=True
        )
        return [serialize_job(job) for job in sorted_jobs]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return serialize_job(job)


@app.post("/api/datasets")
def create_dataset_job(request: DatasetRequest, background_tasks: BackgroundTasks):
    def action(job: Job) -> dict[str, Any]:
        job.progress = 55
        job.message = "Copying images into train and validation folders"
        save_job(job)
        summary = create_dataset(
            Path(request.input_dir),
            Path(request.output_dir),
            request.val_ratio,
            request.seed,
        )
        return asdict(summary)

    job = create_job("dataset", "Create dataset split")
    background_tasks.add_task(run_job, job.id, action)
    return serialize_job(job)


@app.post("/api/training")
def create_training_job(request: TrainingRequest, background_tasks: BackgroundTasks):
    def action(job: Job) -> dict[str, Any]:
        job.progress = 45
        job.message = "Launching OpenCD training"
        save_job(job)
        train(Path(request.config_path), Path(request.work_dir))
        return {"config_path": request.config_path, "work_dir": request.work_dir}

    job = create_job("training", "Train change detection model")
    background_tasks.add_task(run_job, job.id, action)
    return serialize_job(job)


@app.post("/api/inference")
def create_inference_job(request: InferenceRequest, background_tasks: BackgroundTasks):
    def action(job: Job) -> dict[str, Any]:
        job.progress = 45
        job.message = "Launching OpenCD inference"
        save_job(job)
        infer(Path(request.model_path), Path(request.input_dir), Path(request.output_dir))
        return {
            "model_path": request.model_path,
            "input_dir": request.input_dir,
            "output_dir": request.output_dir,
        }

    job = create_job("inference", "Run change detection inference")
    background_tasks.add_task(run_job, job.id, action)
    return serialize_job(job)


@app.post("/detect")
def detect(
    input_path: str,
    background_tasks: BackgroundTasks,
    model_path: Optional[str] = None,
):
    """Backward-compatible endpoint for running inference."""
    input_file = Path(input_path)
    request = InferenceRequest(
        model_path=model_path or "model.pth",
        input_dir=str(input_file.parent),
        output_dir="predictions",
    )
    return create_inference_job(request, background_tasks)


def is_opencd_available() -> bool:
    try:
        import opencd  # noqa: F401
    except ImportError:
        return False
    return True


if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
