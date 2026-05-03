# UAV Change Detection Platform

This project aims to build a change detection platform based on [OpenCD](https://github.com/opendatalab/OpenCD). It provides tools for dataset creation, model training, model inference, and a modern browser dashboard.

## Repository Structure

- `dataset/` – scripts to prepare training datasets.
- `training/` – utilities for training change detection models with OpenCD.
- `inference/` – scripts to run trained models for change detection.
- `backend/` – a FastAPI application exposing API endpoints and serving the dashboard.
- `frontend/` – a modern static dashboard for dataset, training, inference, and job monitoring.

## Development

Install the API dependencies first:
```bash
python3 -m pip install -r requirements.txt
```

1. **Dataset Preparation**
   ```bash
   python dataset/create_dataset.py <raw_images_dir> <output_dataset_dir> [--val-ratio 0.2]
   ```
   Copies images to train/val splits for quick experimentation.
2. **Model Training**
   ```bash
   python training/train_model.py <config.py> <work_dir>
   ```
   Requires [OpenCD](https://github.com/opendatalab/OpenCD) to be installed.
3. **Inference**
   ```bash
   python inference/run_inference.py <model.pth> <input_dir> <pred_dir>
   ```
   Also relies on OpenCD for running the inference pipeline.
4. **Backend and Dashboard**
   ```bash
   uvicorn backend.main:app --reload
   ```

   Open `http://127.0.0.1:8000` to use the dashboard. API endpoints are available under `/api/*`.

## Dashboard Features

- Persistent job history stored in `.uav_platform/jobs.json`.
- Job metrics, live polling, individual job deletion, and one-click history clearing.
- Demo dataset generation from the dashboard for a quick successful dataset split.
- OpenCD readiness display with clear failed task messages when training or inference dependencies are unavailable.

OpenCD is still required for actual model training and inference. If OpenCD is not installed, those jobs are captured as failed tasks with a clear error message.
