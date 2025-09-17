# Hugging Face Space Skeleton

This folder contains a reference implementation of a Hugging Face Space that turns a bundle of sparse photographs into a previewable 3D reconstruction. It assumes the heavy lifting is performed by COLMAP, Nerfstudio, and 3D Gaussian Splatting.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Gradio application that orchestrates preprocessing, reconstruction backends, and artifact export. |
| `requirements.txt` | Python dependencies installed by the Space runtime. |
| `packages.txt` | `apt` packages Hugging Face installs before pip dependencies (COLMAP prerequisites). |
| `assets/demo.zip` | Drop a miniature dataset here for automated smoke tests (optional). |
| `external/gaussian-splatting/` | (Optional) place the upstream Gaussian Splatting repo here if you want to expose that backend. |

## Running locally

```bash
python -m venv .venv               # optional but recommended
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional: only if you want Gaussian Splatting available
git clone https://github.com/graphdeco-inria/gaussian-splatting.git external/gaussian-splatting

# Launch the interface
python app.py
```

Open http://localhost:7860 to access the Gradio UI. Override `HF3D_OUTPUT_ROOT` to choose where run artifacts are stored and `GAUSSIAN_SPLATTING_ROOT` if the repository lives somewhere else on disk.

## Deploying to Hugging Face Spaces

1. Create a new Space with the **Gradio** SDK and GPU hardware.
2. Upload the contents of this folder to the Space repository (or set it as a Git submodule).
3. If you modify dependencies, rebuild the Space and monitor logs for compilation errors (Gaussian Splatting requires CUDA).
4. Enable persistent storage to cache COLMAP databases, Gaussian point clouds, and Nerfstudio checkpoints.

See the root documentation for more context and pipeline details.
