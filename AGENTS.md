# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

This is a UAV Change Detection Platform based on OpenCD. It contains:
- **backend/** — FastAPI app serving the `/detect` endpoint
- **inference/** — CLI wrapper for OpenCD inference
- **training/** — CLI wrapper for OpenCD training
- **dataset/** — Dataset preparation scripts (train/val splitting)
- **frontend/** — Placeholder for a Vue 3 web application (not yet scaffolded)

### Running the Backend

```bash
cd /workspace
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend must be started from the workspace root (`/workspace`) because `backend/main.py` imports from `inference.run_inference`. The `--reload` flag enables hot-reloading during development.

### Key Gotchas

- **No `__init__.py` files in the original repo**: The `backend/`, `inference/`, `training/`, and `dataset/` directories need `__init__.py` files for Python package imports to work. These were added as part of the environment setup.
- **OpenCD is not installed**: The `inference/run_inference.py` and `training/train_model.py` modules check for OpenCD at runtime and raise `RuntimeError` if it's missing. The FastAPI `/detect` endpoint catches this and returns `{"error": "..."}`. This is expected behavior without a GPU/CUDA environment.
- **`dataset/create_dataset.py` uses only stdlib**: No extra dependencies needed for dataset preparation.
- **No automated tests or lint config exist** in this codebase yet.

### Dependencies

See `requirements.txt` for Python dependencies. Install with:
```bash
pip install -r requirements.txt
```
