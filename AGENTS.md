# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

This is a **UAV Change Detection Platform** built on top of OpenCD. It consists of:
- **Backend**: FastAPI app (`backend/main.py`) serving the `/detect` endpoint
- **Dataset tools**: Standalone script (`dataset/create_dataset.py`) for train/val splits
- **Training/Inference**: Scripts wrapping OpenCD CLI tools (require OpenCD + PyTorch)
- **Frontend**: Placeholder (not yet implemented)

### Running the Backend

```bash
cd /workspace
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend imports from `inference.run_inference`, so it must be run from the workspace root for Python path resolution to work correctly.

### Key Gotchas

1. **OpenCD is NOT installed** in the dev environment (requires PyTorch/GPU). The `/detect` endpoint will return an error about missing OpenCD, which is expected behavior for local dev without GPU.
2. **No `__init__.py` files existed originally** — they were added to make Python imports work correctly (`backend/`, `inference/`, `training/`, `dataset/`).
3. **The frontend does not exist yet** — only a placeholder README. No npm/node setup is needed.
4. **`ruff format --check .`** shows two files (`dataset/create_dataset.py`, `training/train_model.py`) need reformatting, but they are part of the original skeleton code.

### Linting and Testing

```bash
ruff check .          # lint check
ruff format --check . # format check
pytest                # run tests (currently no tests exist)
```

### Dependencies

Runtime: `requirements.txt` (fastapi, uvicorn, httpx)
Dev: `requirements-dev.txt` (adds ruff, pytest, httpx)
