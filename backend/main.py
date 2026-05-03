"""FastAPI application for the UAV change detection platform."""

from __future__ import annotations

import json
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
from training.train_model import train, train_demo

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / ".uav_platform"
JOBS_FILE = DATA_DIR / "jobs.json"
DEMO_RAW_DIR = DATA_DIR / "demo_raw"
DEMO_OUTPUT_DIR = DATA_DIR / "demo_processed"

TaskStatus = Literal["queued", "running", "completed", "failed"]
TaskKind = Literal["dataset", "training", "inference", "pipeline"]


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


class DemoDatasetRequest(BaseModel):
    image_count: int = Field(8, ge=2, le=200)
    output_dir: str | None = Field(None, examples=[".uav_platform/demo_processed"])
    val_ratio: float = Field(0.25, ge=0, le=1)
    seed: int | None = Field(42, examples=[42])


class DemoPipelineRequest(DemoDatasetRequest):
    work_dir: str | None = Field(None, examples=[".uav_platform/demo_run"])
    prediction_dir: str | None = Field(None, examples=[".uav_platform/demo_predictions"])


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

jobs_lock = Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_job(job: Job) -> dict[str, Any]:
    return asdict(job)


def load_jobs() -> dict[str, Job]:
    if not JOBS_FILE.exists():
        return {}
    try:
        raw_jobs = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    loaded: dict[str, Job] = {}
    for raw_job in raw_jobs:
        try:
            job = Job(**raw_job)
        except TypeError:
            continue
        if job.status in {"queued", "running"}:
            job.status = "failed"
            job.progress = 100
            job.message = "Server restarted before this task completed"
            job.updated_at = now_iso()
        loaded[job.id] = job
    return loaded


def persist_jobs_unlocked() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = [serialize_job(job) for job in jobs.values()]
    temporary_file = JOBS_FILE.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(JOBS_FILE)


def save_job(job: Job) -> None:
    job.updated_at = now_iso()
    with jobs_lock:
        jobs[job.id] = job
        persist_jobs_unlocked()


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


jobs: dict[str, Job] = load_jobs()


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
        "storage": str(JOBS_FILE),
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


@app.delete("/api/jobs")
def clear_jobs():
    with jobs_lock:
        jobs.clear()
        persist_jobs_unlocked()
    return {"message": "All jobs cleared"}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        del jobs[job_id]
        persist_jobs_unlocked()
    return {"message": "Job deleted", "job_id": job_id}


@app.post("/api/demo-dataset")
def prepare_demo_dataset(request: DemoDatasetRequest):
    demo = build_demo_images(request.image_count)
    output_dir = Path(request.output_dir) if request.output_dir else DEMO_OUTPUT_DIR
    return {
        "input_dir": str(demo),
        "output_dir": str(output_dir),
        "image_count": request.image_count,
        "val_ratio": request.val_ratio,
        "seed": request.seed,
    }


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


@app.post("/api/demo-pipeline")
def create_demo_pipeline_job(
    request: DemoPipelineRequest,
    background_tasks: BackgroundTasks,
):
    def action(job: Job) -> dict[str, Any]:
        raw_dir = build_demo_images(request.image_count)
        dataset_dir = Path(request.output_dir) if request.output_dir else DEMO_OUTPUT_DIR
        work_dir = Path(request.work_dir) if request.work_dir else DATA_DIR / "demo_run"
        prediction_dir = (
            Path(request.prediction_dir)
            if request.prediction_dir
            else DATA_DIR / "demo_predictions"
        )

        job.progress = 25
        job.message = "Creating demo dataset split"
        save_job(job)
        dataset_summary = create_dataset(
            raw_dir,
            dataset_dir,
            request.val_ratio,
            request.seed,
        )

        job.progress = 55
        job.message = "Training demo model artifact"
        save_job(job)
        model_summary = train_demo(Path("demo_config.py"), work_dir, dataset_dir)

        job.progress = 82
        job.message = "Generating demo prediction masks"
        save_job(job)
        inference_summary = infer(
            Path(model_summary.model_path),
            dataset_dir / "val",
            prediction_dir,
            demo_mode=True,
        )
        return {
            "mode": "demo",
            "dataset": asdict(dataset_summary),
            "training": asdict(model_summary),
            "inference": inference_summary,
        }

    job = create_job("pipeline", "Run complete demo pipeline")
    background_tasks.add_task(run_job, job.id, action)
    return serialize_job(job)


@app.post("/api/training")
def create_training_job(request: TrainingRequest, background_tasks: BackgroundTasks):
    def action(job: Job) -> dict[str, Any]:
        job.progress = 45
        job.message = "Launching training"
        save_job(job)
        config_path = Path(request.config_path)
        work_dir = Path(request.work_dir)
        if is_opencd_available():
            train(config_path, work_dir)
            return {
                "mode": "opencd",
                "config_path": request.config_path,
                "work_dir": request.work_dir,
            }
        return asdict(train_demo(config_path, work_dir))

    job = create_job("training", "Train change detection model")
    background_tasks.add_task(run_job, job.id, action)
    return serialize_job(job)


@app.post("/api/inference")
def create_inference_job(request: InferenceRequest, background_tasks: BackgroundTasks):
    def action(job: Job) -> dict[str, Any]:
        job.progress = 45
        job.message = "Launching inference"
        save_job(job)
        return infer(
            Path(request.model_path),
            Path(request.input_dir),
            Path(request.output_dir),
            demo_mode=not is_opencd_available(),
        )

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


def build_demo_images(image_count: int) -> Path:
    DEMO_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for stale_image in DEMO_RAW_DIR.glob("*"):
        if stale_image.is_file():
            stale_image.unlink()
    for index in range(image_count):
        image_path = DEMO_RAW_DIR / f"demo_uav_{index + 1:03d}.jpg"
        image_path.write_bytes(b"demo-uav-image")
    return DEMO_RAW_DIR


if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
