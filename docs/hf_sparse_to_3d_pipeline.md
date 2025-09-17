# Sparse-Image to 3D Reconstruction Pipeline for Hugging Face Spaces

This guide walks through the full lifecycle required to turn sparse, non-overlapping photographs into a georeferenced 3D asset hosted on a Hugging Face Space. It assumes you are preparing material for GIS-focused urban-planning research and want reproducible steps from dataset assembly to a working web demo.

> **Key idea:** Split the problem into (1) structure-from-motion (SfM) with COLMAP, (2) neural reconstruction with either Nerfstudio (NeRF), 3D Gaussian Splatting, or a hybrid, (3) geospatial post-processing, and (4) packaging the experience into a GPU-backed Space powered by Gradio.

---

## 1. Project goals & success criteria

1. **Robust on sparse imagery:** Inputs may have large baselines and limited overlap, so the pipeline must use reliable feature matching (COLMAP) and optionally auxiliary priors (VGGT depth).
2. **GIS friendly outputs:** Deliver textured meshes (`.glb`/`.obj`), point clouds (`.ply`/`.las`), and optionally rasterized DSM/DTM layers that can be ingested by QGIS or ArcGIS Pro.
3. **Interactive Hugging Face demo:** Provide a web UI that accepts images, runs the reconstruction pipeline (either fully on Space hardware or as an orchestrator for offline runs), and previews the result.
4. **Extensible research harness:** Make it easy to swap in cutting-edge CVPR 2024+ methods (e.g., PointNeRF variants, 3D-R1 for VLM-guided priors) as your PhD evolves.

---

## 2. Core toolkit overview

| Stage | Recommended tooling | Why it matters |
| --- | --- | --- |
| Image calibration & SfM | [COLMAP](https://github.com/colmap/colmap) | De facto standard for camera pose estimation and sparse/dense reconstruction. |
| Depth priors | [VGGT / PointMap](https://vgg-t.github.io/) | Helps regularize sparse scenes by providing depth & normal predictions that COLMAP can fuse. |
| Neural rendering | [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) with `instant-ngp` or `nerfacto` configs | Streamlined training scripts, evaluation, and export to mesh/point clouds. |
| Fast point-based models | [3D Gaussian Splatting (Inria)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | Faster convergence than NeRF, works well with limited views when initialized from COLMAP cameras. |
| Indoor-specific | [Fast3R (Meta)](https://github.com/facebookresearch/fast3r) | Handles Manhattan-world indoor scenes if you target building interiors. |
| Language priors | [3D-R1](https://github.com/AIGeeksGroup/3D-R1) | Use if you want to integrate VLM guidance or textual semantics. |

Feel free to incorporate other CVPR 2023/2024 papers—this pipeline is modular.

---

## 3. Local development environment

1. **Hardware**: CUDA-capable GPU (>=12 GB VRAM recommended). CPU-only runs are possible for COLMAP but neural training will be slow.
2. **System packages** (Ubuntu 22.04 example):
   ```bash
   sudo apt update
   sudo apt install -y build-essential cmake git wget ninja-build \
       libboost-all-dev libeigen3-dev libfreeimage-dev libmetis-dev \
       libgoogle-glog-dev libgflags-dev libglew-dev qtbase5-dev
   ```
3. **Python environment**:
   ```bash
   conda create -n hf-3d python=3.10
   conda activate hf-3d
   pip install --upgrade pip
   ```
4. **Install COLMAP** (choose one):
   - Build from source following the [official instructions](https://colmap.github.io/install.html).
   - Or `conda install -c conda-forge colmap` if using Conda-forge packages.
5. **Install Nerfstudio**:
   ```bash
   pip install nerfstudio
   ns-install-cli
   ```
6. **Install 3D Gaussian Splatting** (requires CUDA build):
   ```bash
   git clone https://github.com/graphdeco-inria/gaussian-splatting.git
   cd gaussian-splatting
   pip install -r requirements.txt
   python setup.py build_ext --inplace
   ```
7. **Optional: VGGT depth & PointMap**:
   ```bash
   git clone https://github.com/nianticlabs/pointmap.git
   cd pointmap
   pip install -r requirements.txt
   ```
8. **Validate** by running small demo datasets provided by Nerfstudio (`ns-download-data --dataset nerfstudio --capture nerfstudio --output-dir data/nerfstudio`).

> Tip: Use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [mamba](https://mamba.readthedocs.io/) to isolate dependencies between COLMAP, Nerfstudio, and the Gaussian Splatting CUDA extensions.

---

## 4. Data preparation workflow

1. **Collect imagery**
   - Capture JPEG/PNG photos with consistent exposure where possible.
   - Record GPS/IMU metadata if available (smartphone EXIF is fine). Store these in a CSV for later georeferencing.
   - For aerial photogrammetry, maintain at least ~60% forward overlap even if lateral overlap is low.
2. **Organize dataset**
   ```text
   dataset/
     images/
       img_0001.jpg
       img_0002.jpg
     meta.csv            # optional per-image metadata
   ```
3. **Run COLMAP SfM** (automatic reconstruction):
   ```bash
   mkdir -p outputs/colmap
   colmap feature_extractor \
       --database_path outputs/colmap/database.db \
       --image_path dataset/images \
       --SiftExtraction.use_gpu 1

   colmap exhaustive_matcher \
       --database_path outputs/colmap/database.db \
       --SiftMatching.use_gpu 1

   colmap mapper \
       --database_path outputs/colmap/database.db \
       --image_path dataset/images \
       --output_path outputs/colmap/sparse

   colmap image_undistorter \
       --image_path dataset/images \
       --input_path outputs/colmap/sparse/0 \
       --output_path outputs/colmap/dense \
       --output_type COLMAP
   ```
4. **Depth fusion (optional but recommended)**
   - Use VGGT to predict per-image depth maps:
     ```bash
     python tools/infer_vggt_depth.py \
         --images dataset/images \
         --output outputs/vggt/depths
     ```
   - Fuse VGGT priors in COLMAP's dense reconstruction:
     ```bash
     colmap patch_match_stereo \
         --workspace_path outputs/colmap/dense \
         --PatchMatchStereo.geom_consistency true \
         --PatchMatchStereo.depth_prior_path outputs/vggt/depths

     colmap stereo_fusion \
         --workspace_path outputs/colmap/dense \
         --output_path outputs/colmap/fused.ply
     ```
5. **Inspect coverage** using `colmap gui` or Meshlab. Re-capture imagery if coverage gaps exist.
6. **Export camera poses** for downstream models:
   ```bash
   colmap model_converter \
       --input_path outputs/colmap/sparse/0 \
       --output_path outputs/nerfstudio \
       --output_type NERFSTUDIO
   ```

---

## 5. Neural reconstruction backends

### 5.1 Nerfstudio (NeRF)
1. **Prepare Nerfstudio dataset**:
   ```bash
   ns-process-data images --data data/dataset --output-dir data/nerfstudio-dataset
   ```
2. **Train** (choose configuration based on sparsity):
   ```bash
   ns-train nerfacto \
       --data data/nerfstudio-dataset \
       --pipeline.model.depth-importance 0.3 \
       --pipeline.datamanager.camera-optimizer.mode off
   ```
   - Use `--pipeline.model.background-color random` to improve robustness for outdoor imagery.
   - For extremely sparse captures, try `ns-train splatfacto` which mixes point splatting with NeRF.
3. **Evaluate**:
   ```bash
   ns-eval --load-config outputs/nerfacto/2024-xx-xx/config.yml
   ```
4. **Export**:
   ```bash
   ns-export poisson \
       --load-config outputs/nerfacto/2024-xx-xx/config.yml \
       --output-path exports/mesh

   ns-export gaussian-splat \
       --load-config outputs/nerfacto/2024-xx-xx/config.yml \
       --output-path exports/gaussians
   ```

### 5.2 3D Gaussian Splatting
1. **Convert COLMAP cameras** to the expected format:
   ```bash
   python convert.py -s outputs/colmap/dense -o data/gaussian
   ```
   (Use the conversion script provided in the Gaussian Splatting repository.)
2. **Train / optimize**:
   ```bash
   python train.py \
       -s data/gaussian \
       -m outputs/gaussian \
       --iterations 7000 \
       --resolution 2
   ```
3. **Export** Gaussian cloud as `.ply` or `.splat` and convert to mesh with [Gaussian Surfels](https://github.com/antimatter15/splat) if a watertight surface is needed.

### 5.3 Hybrid or alternative models
- **Fast3R**: Useful for structured indoor scans; follow repo instructions and feed COLMAP poses.
- **3D-R1**: Combine textual prompts (e.g., building materials) with geometry to hallucinate missing views.
- **Recent CVPR papers**: Keep a `research/notes.md` log and plug new methods as additional backends in the Space (see Section 8.4).

---

## 6. Post-processing & GIS integration

1. **Scale & georeference**
   - Align the reconstructed model with ground control points (GCPs) using tools like [OpenSfM GeoAligner](https://github.com/mapillary/OpenSfM) or `pytransform3d`.
   - Convert from local COLMAP coordinates to EPSG codes relevant to your study area using `pyproj`.
2. **Clean geometry**
   - Use [Meshlab](https://www.meshlab.net/) or [Blender](https://www.blender.org/) for decimation, hole filling, and texturing.
   - Export `glb` for real-time viewers and `las`/`laz` for point clouds.
3. **Generate rasters**
   - Rasterize point clouds with PDAL to produce DSM/DTM/NDVI layers:
     ```bash
     pdal pipeline pipelines/dsm.json
     ```
4. **Metadata packaging**
   - Create a `metadata.json` storing CRS, scale, original photo IDs, and licensing info for reproducibility.

---

## 7. Hugging Face Space deployment plan

### 7.1 Repository layout
```
.
├── hf_space/
│   ├── app.py              # Gradio UI + orchestration logic
│   ├── requirements.txt    # Python dependencies installed by the Space runtime
│   ├── packages.txt        # apt packages needed for COLMAP and OpenGL support
│   ├── README.md           # Usage instructions shown on Space page
│   ├── assets/
│   │   └── demo.zip        # (Optional) sample dataset for smoke testing
│   └── external/
│       └── gaussian-splatting/   # (Optional) clone of the upstream repository
└── docs/
    └── ...
```

### 7.2 Space hardware & settings
- **Hardware**: Request at least an `A10G` or `A100` GPU Space. COLMAP + Nerfstudio benefit from >12 GB VRAM.
- **Runtime**: Stick with the default **Gradio** Space SDK. Dependencies install automatically from `requirements.txt`, while `packages.txt` pulls in the system-level libraries COLMAP needs (OpenGL, Eigen, Boost, etc.).
- **External repositories**: If you plan to expose the Gaussian Splatting backend, clone the upstream repository into `hf_space/external/gaussian-splatting` or set the `GAUSSIAN_SPLATTING_ROOT` environment variable so the app can find it at runtime.
- **Persistent storage**: Enable the `Persistent storage` option so trained models and cached COLMAP databases survive restarts.
- **Secrets**: Store API keys (if any) using the Space Secrets feature. Not strictly needed unless you access private datasets.

### 7.3 CI / precompute strategy
- Running the entire pipeline on every user upload may be too slow. Two deployment options:
  1. **Interactive preview (recommended)**: Run COLMAP + Gaussian Splatting on a subset of uploaded images (or downscaled versions) directly in the Space, returning a low-res preview + instructions to download a zipped workspace for offline refinement.
  2. **Asynchronous jobs**: Use [Spaces Webhooks](https://huggingface.co/docs/hub/spaces-webhooks) to forward uploads to a private GPU server, then notify the user via email/slack when high-quality outputs are ready.

### 7.4 Integrating new backends
- The `ReconstructionRunner` class in `hf_space/app.py` exposes a `register_backend` method. Plug any new CVPR models by providing a callable that accepts the workspace path and returns the exported artifact.
- Keep each backend self-contained (install requirements, compile CUDA extensions) to avoid runtime conflicts.

---

## 8. Running the Hugging Face Space locally

1. **Clone this repository** and enter `hf_space/`.
2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install prerequisites**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   If you intend to exercise the Gaussian Splatting path, clone its repository so the app can access it:
   ```bash
   git clone https://github.com/graphdeco-inria/gaussian-splatting.git external/gaussian-splatting
   ```
4. **Launch the Gradio app**:
   ```bash
   python app.py
   ```
   By default, the interface listens on `http://127.0.0.1:7860`. Set `HF3D_OUTPUT_ROOT=/path/to/runs` to control where intermediate artifacts are written, and `GAUSSIAN_SPLATTING_ROOT` if you cloned the repository elsewhere.
5. **Upload test images** (drag & drop zipped images or use the sample asset).
6. **Push to Hugging Face**:
   ```bash
   huggingface-cli login
   huggingface-cli repo create username/urban-3d-space --type space --sdk gradio
   git remote add space https://huggingface.co/spaces/username/urban-3d-space
   git push space main
   ```
7. **Monitor** the build logs on the Space page and resolve missing dependencies if necessary.

---

## 9. Operational checklist

- [ ] Validate dataset ingestion on at least two capture types (street-level photos and UAV imagery).
- [ ] Benchmark inference time for Nerfstudio vs Gaussian Splatting on your Space hardware.
- [ ] Enable persistent storage and test restart resilience.
- [ ] Document CRS conversions and maintain a changelog for your advisor.
- [ ] Set up automated smoke tests (e.g., run COLMAP on a 3-image sample) using GitHub Actions or Spaces CI.

---

## 10. Further reading & research leads

- CVPR 2024 highlights: check `HoloDiffusion`, `LGM`, `Street Gaussians` for urban-scale reconstructions.
- Explore `sat2pc` or `CityNeRF` papers if you plan to merge satellite imagery with ground photos.
- For GIS integration, read the [OGC 3D Tiles specification](https://www.ogc.org/standard/3dtiles/) and consider exporting to Cesium for web-based city planning demos.
- Keep track of [Nerfstudio releases](https://github.com/nerfstudio-project/nerfstudio/releases) for new training configs tailored to sparse captures.

---

## 11. Support matrix

| Component | Tested locally | Runs on Hugging Face GPU Space | Notes |
| --- | --- | --- | --- |
| COLMAP SfM | ✅ | ✅ (via `packages.txt`) | Requires OpenGL runtime; ensure the Space hardware supports CUDA acceleration. |
| Nerfstudio (nerfacto) | ✅ | ✅ | Install via pip; set `--max-num-iterations` to cap runtime on shared hardware. |
| 3D Gaussian Splatting | ✅ | ⚠️ | CUDA extensions compile during build; may need to pre-build wheels to reduce Space build time. |
| VGGT depth | ⚠️ | ⚠️ | Heavy ViT models; download checkpoints during build and store in persistent volume. |
| Fast3R | ⚠️ | ⚠️ | Useful only for indoor scenes; optional. |

Legend: ✅ = recommended, ⚠️ = optional/experimental, ❌ = not supported.

---

By following this guide and adapting the sample Space in this repository, you can deliver a production-quality 3D reconstruction demo for your GIS-oriented PhD application while leaving room to incorporate the latest research ideas.
