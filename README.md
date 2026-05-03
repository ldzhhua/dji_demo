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
   Real training uses the OpenCD repository entry point `tools/train.py`. Set `OPENCD_ROOT=/path/to/open-cd` so the platform can locate it. For local smoke tests, add `--demo` to create deterministic demo model artifacts.
3. **Inference**
   ```bash
   python inference/run_inference.py <config.py> <model.pth> <input_dir> <pred_dir>
   ```
   Real inference uses `opencd.apis.OpenCDInferencer`. Put A/B image pairs in subfolders named `A` and `B`, or use filenames ending in `_A` and `_B`. For local smoke tests, add `--demo-mode` to generate deterministic demo prediction masks.
4. **Backend and Dashboard**
   ```bash
   uvicorn backend.main:app --reload
   ```

   Open `http://127.0.0.1:8000` to use the dashboard. API endpoints are available under `/api/*`.

## Dashboard Features

- Persistent job history stored in `.uav_platform/jobs.json`.
- Job metrics, live polling, individual job deletion, and one-click history clearing.
- Demo dataset generation from the dashboard for a quick successful dataset split.
- Complete demo pipeline that runs dataset preparation, demo training, and demo inference end to end without OpenCD.
- OpenCD readiness display with real training/inference forms for config, checkpoint, classes, and palette options.

OpenCD is still required for actual model training and inference. If OpenCD is not installed, use the dashboard's "一键完整跑通" button to run a full local demo, or submit real OpenCD jobs and see clear failed task messages.
