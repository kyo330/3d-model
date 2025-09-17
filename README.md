# 3D Model Reconstruction for GIS

This repository documents a complete workflow for turning sparse, non-overlapping photographs into interactive 3D assets that can be hosted on a [Hugging Face Space](https://huggingface.co/spaces). It focuses on pipelines that are appropriate for GIS and urban-planning research, integrating structure-from-motion (SfM), neural rendering (NeRF), and Gaussian Splatting approaches.

## How to use this repository

1. **Study the end-to-end plan** in [`docs/hf_sparse_to_3d_pipeline.md`](docs/hf_sparse_to_3d_pipeline.md). It explains the tooling choices (COLMAP, Nerfstudio, 3D Gaussian Splatting, VGGT depth), provides step-by-step commands, and highlights GIS-specific post-processing.
2. **Start from the Hugging Face Space skeleton** under [`hf_space/`](hf_space/). The folder contains a production-ready `gradio` application and dependency manifests that you can adapt to your project and deploy directly on Hugging Face without building a Docker image.
3. **Iterate on the pipeline** by wiring additional reconstruction backends, adding evaluation hooks, or integrating geospatial metadata exports as required by your PhD project.

The repository is intentionally modular so that you can swap in newer research ideas (for example recent CVPR papers) without rewriting the Space front-end.
