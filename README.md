# UAV Change Detection Platform

This project aims to build a change detection platform based on [OpenCD](https://github.com/opendatalab/OpenCD). It will provide tools for dataset creation, model training, and model inference using Python 3.10+ along with a Vue 3 frontend.

## Repository Structure

- `dataset/` – scripts to prepare training datasets.
- `training/` – utilities for training change detection models with OpenCD.
- `inference/` – scripts to run trained models for change detection.
- `backend/` – a simple FastAPI application exposing API endpoints.
- `frontend/` – placeholder for the Vue 3 web application.

## Development

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
4. **Backend**
   ```bash
   uvicorn backend.main:app --reload
   ```
5. **Frontend**
   See `frontend/README.md` for instructions on creating the Vue 3 project.

This repository currently provides skeleton code to get started. Detailed implementation of data processing, training logic, and inference integration with OpenCD is left as future work.
