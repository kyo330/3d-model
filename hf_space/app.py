from __future__ import annotations

import datetime as dt
import io
import json
import os
import shutil
import subprocess
import textwrap
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import gradio as gr
from PIL import Image


def _run_command(command: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Execute a shell command and capture combined stdout/stderr."""
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process.returncode, process.stdout


@dataclass
class Backend:
    name: str
    description: str
    runner: Callable[[Path, Path, Optional[Dict[str, Path]], int], Tuple[Path, List[str]]]


class ReconstructionRunner:
    """Coordinate preprocessing, COLMAP, and neural backends."""

    def __init__(self, output_root: Optional[Path] = None) -> None:
        root = output_root or Path(os.environ.get("HF3D_OUTPUT_ROOT", "/tmp/hf_3d_runs"))
        root.mkdir(parents=True, exist_ok=True)
        self.output_root = root
        self.backends: Dict[str, Backend] = {}
        self._register_default_backends()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def available_methods(self) -> List[str]:
        return list(self.backends.keys())

    def describe_backend(self, name: str) -> str:
        backend = self.backends.get(name)
        return backend.description if backend else ""

    def run(
        self,
        uploads: Iterable[Any],
        method: str,
        max_resolution: int,
        skip_colmap: bool,
    ) -> Tuple[str, Optional[Path]]:
        logs: List[str] = []
        timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        workspace = self.output_root / f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
        dataset_root = workspace / "dataset"
        images_dir = dataset_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        logs.append(f"Workspace initialized at {workspace}")

        try:
            ingest_count = self._ingest_uploads(uploads, images_dir, max_resolution)
        except Exception as exc:  # noqa: BLE001 - top-level guard for user feedback
            logs.append(f"[ERROR] Failed to ingest inputs: {exc}")
            return "\n".join(logs), None

        if ingest_count == 0:
            logs.append("[ERROR] No images detected in upload. Provide JPG/PNG files or a ZIP archive.")
            return "\n".join(logs), None

        logs.append(f"Ingested {ingest_count} image(s). Max resolution capped at {max_resolution}px")

        colmap_outputs: Optional[Dict[str, Path]] = None
        if skip_colmap:
            logs.append("Skipping COLMAP as requested. Downstream models must rely on precomputed poses.")
        else:
            try:
                colmap_outputs, colmap_logs = self._run_colmap(images_dir, workspace / "colmap", max_resolution)
                logs.extend(colmap_logs)
            except FileNotFoundError as exc:
                logs.append(
                    textwrap.dedent(
                        f"""
                        [ERROR] Required binary `{exc}` was not found. Ensure COLMAP is installed or set
                        `skip_colmap=True` if you plan to upload precomputed camera poses.
                        """
                    ).strip()
                )
                return "\n".join(logs), None
            except RuntimeError as exc:
                logs.append(str(exc))
                return "\n".join(logs), None

        backend = self.backends.get(method)
        if not backend:
            logs.append(f"[ERROR] Unknown backend '{method}'. Available options: {', '.join(self.available_methods())}")
            return "\n".join(logs), None

        try:
            artifact_path, backend_logs = backend.runner(workspace, dataset_root, colmap_outputs, max_resolution)
            logs.extend(backend_logs)
        except Exception as exc:  # noqa: BLE001 - propagate details to UI
            logs.append(f"[ERROR] Backend '{method}' failed: {exc}")
            return "\n".join(logs), None

        logs.append(f"Artifacts packaged at {artifact_path}")
        return "\n".join(logs), artifact_path

    # ------------------------------------------------------------------
    # Backend registration
    # ------------------------------------------------------------------
    def register_backend(self, backend: Backend) -> None:
        self.backends[backend.name] = backend

    def _register_default_backends(self) -> None:
        self.register_backend(
            Backend(
                name="Nerfstudio (NeRF)",
                description=(
                    "Optimizes a NeRF with the nerfacto recipe, exports a Poisson surface mesh, and packs all outputs "
                    "(config, checkpoints, mesh, transforms.json) into a ZIP archive."
                ),
                runner=self._run_nerfstudio,
            )
        )
        self.register_backend(
            Backend(
                name="3D Gaussian Splatting",
                description=(
                    "Uses the Inria Gaussian Splatting reference implementation initialized from COLMAP cameras. "
                    "Returns the optimized Gaussian point cloud and training logs."
                ),
                runner=self._run_gaussian_splatting,
            )
        )

    # ------------------------------------------------------------------
    # Input ingestion helpers
    # ------------------------------------------------------------------
    def _ingest_uploads(self, uploads: Iterable[Any], images_dir: Path, max_resolution: int) -> int:
        metadata: List[Dict[str, object]] = []
        count = 0
        for item in uploads:
            if not item:
                continue
            src_path = Path(getattr(item, "name", getattr(item, "path", "")))
            if not src_path.exists():
                # Gradio may store temp files in `.name`; fallback to `.path` when available
                if hasattr(item, "path"):
                    src_path = Path(item.path)
            if not src_path.exists():
                continue

            if zipfile.is_zipfile(src_path):
                with zipfile.ZipFile(src_path, "r") as archive:
                    for member in archive.namelist():
                        lower = member.lower()
                        if lower.endswith((".jpg", ".jpeg", ".png")):
                            data = archive.read(member)
                            image = Image.open(io.BytesIO(data))
                            dest = images_dir / Path(member).name
                            self._save_image(image, dest, max_resolution)
                            metadata.append(self._image_metadata(dest, source=str(member)))
                            count += 1
            else:
                image = Image.open(src_path)
                dest = images_dir / src_path.name
                self._save_image(image, dest, max_resolution)
                metadata.append(self._image_metadata(dest, source=str(src_path.name)))
                count += 1

        if metadata:
            dataset_meta = {
                "created_at": dt.datetime.utcnow().isoformat() + "Z",
                "max_resolution": max_resolution,
                "images": metadata,
            }
            meta_path = images_dir.parent / "metadata.json"
            meta_path.write_text(json.dumps(dataset_meta, indent=2))
        return count

    @staticmethod
    def _save_image(image: Image.Image, destination: Path, max_resolution: int) -> None:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, max_resolution / max(width, height))
        if scale < 1.0:
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, quality=95)

    @staticmethod
    def _image_metadata(path: Path, source: str) -> Dict[str, object]:
        with Image.open(path) as image:
            width, height = image.size
        return {
            "filename": path.name,
            "width": width,
            "height": height,
            "source": source,
        }

    # ------------------------------------------------------------------
    # COLMAP integration
    # ------------------------------------------------------------------
    def _run_colmap(self, images_dir: Path, output_dir: Path, max_resolution: int) -> Tuple[Dict[str, Path], List[str]]:
        if shutil.which("colmap") is None:
            raise FileNotFoundError("colmap")

        logs: List[str] = ["Running COLMAP reconstruction…"]
        output_dir.mkdir(parents=True, exist_ok=True)
        database_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        dense_dir = output_dir / "dense"
        sparse_dir.mkdir(exist_ok=True)

        commands = [
            (
                "Feature extraction",
                [
                    "colmap",
                    "feature_extractor",
                    "--database_path",
                    str(database_path),
                    "--image_path",
                    str(images_dir),
                    "--SiftExtraction.use_gpu",
                    "1",
                    "--SiftExtraction.max_image_size",
                    str(max_resolution),
                ],
            ),
            (
                "Exhaustive matcher",
                [
                    "colmap",
                    "exhaustive_matcher",
                    "--database_path",
                    str(database_path),
                    "--SiftMatching.use_gpu",
                    "1",
                ],
            ),
            (
                "Mapper",
                [
                    "colmap",
                    "mapper",
                    "--database_path",
                    str(database_path),
                    "--image_path",
                    str(images_dir),
                    "--output_path",
                    str(sparse_dir),
                ],
            ),
            (
                "Image undistorter",
                [
                    "colmap",
                    "image_undistorter",
                    "--image_path",
                    str(images_dir),
                    "--input_path",
                    str(sparse_dir / "0"),
                    "--output_path",
                    str(dense_dir),
                    "--output_type",
                    "COLMAP",
                ],
            ),
        ]

        for stage, command in commands:
            logs.append(f"\n$ {' '.join(command)}")
            code, output = _run_command(command)
            logs.append(output)
            if code != 0:
                raise RuntimeError(f"[ERROR] COLMAP stage '{stage}' failed with exit code {code}.")

        outputs = {
            "database": database_path,
            "sparse": sparse_dir / "0",
            "dense": dense_dir,
        }
        logs.append("COLMAP completed successfully.")
        return outputs, logs

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------
    def _run_nerfstudio(
        self,
        workspace: Path,
        dataset_root: Path,
        colmap_outputs: Optional[Dict[str, Path]],
        max_resolution: int,
    ) -> Tuple[Path, List[str]]:
        if shutil.which("ns-train") is None:
            raise FileNotFoundError("ns-train")

        logs: List[str] = ["Launching Nerfstudio pipeline…"]
        processed_dir = workspace / "nerfstudio" / "processed"
        runs_dir = workspace / "nerfstudio" / "runs"
        export_dir = workspace / "nerfstudio" / "export"
        processed_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        data_source = dataset_root / "images"
        process_cmd = [
            "ns-process-data",
            "images",
            "--data",
            str(data_source),
            "--output-dir",
            str(processed_dir),
            "--max-num-downscales",
            str(max(1, int(max_resolution / 512))),
        ]
        if colmap_outputs:
            process_cmd.extend(["--skip-colmap"])
            process_cmd.extend(["--colmap-model-path", str(colmap_outputs["sparse"])])

        logs.append(f"\n$ {' '.join(process_cmd)}")
        code, output = _run_command(process_cmd)
        logs.append(output)
        if code != 0:
            raise RuntimeError("ns-process-data failed. See logs above.")

        train_cmd = [
            "ns-train",
            "nerfacto",
            "--data",
            str(processed_dir),
            "--max-num-iterations",
            "3000",
            "--output-dir",
            str(runs_dir),
            "--viewer.quit-on-train-completion",
            "True",
            "--pipeline.model.depth-importance",
            "0.3",
        ]
        logs.append(f"\n$ {' '.join(train_cmd)}")
        code, output = _run_command(train_cmd)
        logs.append(output)
        if code != 0:
            raise RuntimeError("ns-train failed. Consider reducing iterations or verifying GPU availability.")

        configs = sorted(runs_dir.rglob("config.yml"))
        if not configs:
            raise RuntimeError("Unable to locate Nerfstudio config.yml after training.")
        config_path = configs[-1]

        export_cmd = [
            "ns-export",
            "poisson",
            "--load-config",
            str(config_path),
            "--output-path",
            str(export_dir),
        ]
        logs.append(f"\n$ {' '.join(export_cmd)}")
        code, output = _run_command(export_cmd)
        logs.append(output)
        if code != 0:
            raise RuntimeError("ns-export failed. Check above logs for details.")

        mesh_path = export_dir / "mesh.obj"
        artifact_path = workspace / "nerfstudio_result.zip"
        with zipfile.ZipFile(artifact_path, "w") as archive:
            for path in [mesh_path, export_dir / "mesh.mtl", config_path, processed_dir / "transforms.json"]:
                if path.exists():
                    archive.write(path, arcname=path.relative_to(workspace))
            for ckpt in runs_dir.rglob("*.ckpt"):
                archive.write(ckpt, arcname=ckpt.relative_to(workspace))
        logs.append("Nerfstudio export complete.")
        return artifact_path, logs

    def _run_gaussian_splatting(
        self,
        workspace: Path,
        dataset_root: Path,
        colmap_outputs: Optional[Dict[str, Path]],
        max_resolution: int,
    ) -> Tuple[Path, List[str]]:
        default_repo = Path(__file__).resolve().parent / "external" / "gaussian-splatting"
        repo_root = Path(os.environ.get("GAUSSIAN_SPLATTING_ROOT", default_repo))
        convert_script = repo_root / "convert.py"
        train_script = repo_root / "train.py"
        if not convert_script.exists() or not train_script.exists():
            raise FileNotFoundError(
                "Gaussian Splatting repository not found. Clone it to 'external/gaussian-splatting' "
                "or set GAUSSIAN_SPLATTING_ROOT to point at the upstream project."
            )
        if not colmap_outputs:
            raise RuntimeError("Gaussian Splatting requires COLMAP outputs. Disable 'Skip COLMAP'.")

        logs: List[str] = ["Launching 3D Gaussian Splatting pipeline…"]
        gaussian_root = workspace / "gaussian"
        data_dir = gaussian_root / "data"
        model_dir = gaussian_root / "model"
        gaussian_root.mkdir(parents=True, exist_ok=True)

        convert_cmd = [
            "python3",
            str(convert_script),
            "-s",
            str(colmap_outputs["dense"]),
            "-o",
            str(data_dir),
        ]
        logs.append(f"\n$ {' '.join(convert_cmd)}")
        code, output = _run_command(convert_cmd, cwd=repo_root)
        logs.append(output)
        if code != 0:
            raise RuntimeError("Gaussian Splatting conversion failed. Verify COLMAP dense output.")

        train_cmd = [
            "python3",
            str(train_script),
            "-s",
            str(data_dir),
            "-m",
            str(model_dir),
            "--iterations",
            "7000",
            "--resolution",
            str(max(1, max_resolution // 512)),
        ]
        logs.append(f"\n$ {' '.join(train_cmd)}")
        code, output = _run_command(train_cmd, cwd=repo_root)
        logs.append(output)
        if code != 0:
            raise RuntimeError("Gaussian Splatting training failed. See logs for CUDA-related messages.")

        ply_candidates = sorted(model_dir.rglob("*.ply"))
        if not ply_candidates:
            raise RuntimeError("No PLY point cloud found after Gaussian Splatting training.")
        ply_path = ply_candidates[-1]

        artifact_path = workspace / "gaussian_result.zip"
        with zipfile.ZipFile(artifact_path, "w") as archive:
            archive.write(ply_path, arcname=ply_path.relative_to(workspace))
            for log_file in gaussian_root.rglob("*.log"):
                archive.write(log_file, arcname=log_file.relative_to(workspace))
        logs.append("Gaussian Splatting export complete.")
        return artifact_path, logs


# ----------------------------------------------------------------------
# Gradio interface
# ----------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    output_override = os.environ.get("HF3D_OUTPUT_ROOT")
    if output_override:
        output_root = Path(output_override)
    else:
        output_root = Path(__file__).resolve().parent / "runs"
    runner = ReconstructionRunner(output_root=output_root)

    with gr.Blocks(title="Sparse Images to 3D Reconstruction") as demo:
        gr.Markdown(
            textwrap.dedent(
                """
                # Sparse Images ➜ 3D Reconstruction

                Upload a folder or ZIP archive of sparse, non-overlapping photographs. The app will run COLMAP to estimate camera
                poses, then optimize either a Nerfstudio NeRF or a 3D Gaussian Splatting model and return a downloadable artifact.
                Expect several minutes of processing time for high-resolution captures.
                """
            )
        )

        with gr.Row():
            uploads = gr.Files(label="Images or ZIP archive", file_types=["image", ".zip"], file_count="multiple")
            method = gr.Dropdown(
                choices=runner.available_methods(),
                value="Nerfstudio (NeRF)",
                label="Reconstruction backend",
            )

        with gr.Row():
            max_resolution = gr.Slider(
                minimum=512,
                maximum=4096,
                step=256,
                value=2048,
                label="Max processing resolution (pixels)",
            )
            skip_colmap = gr.Checkbox(
                value=False,
                label="Skip COLMAP (use existing poses)",
            )

        default_backend = runner.available_methods()[0] if runner.available_methods() else ""
        backend_description = gr.Markdown(runner.describe_backend(default_backend))
        method.change(
            fn=lambda choice: runner.describe_backend(choice),
            inputs=method,
            outputs=backend_description,
        )
        run_button = gr.Button("Start reconstruction", variant="primary")

        logs = gr.Textbox(label="Pipeline log", lines=20)
        artifact = gr.File(label="Download results")

        def _execute(files: List[Any], backend: str, resolution: int, skip: bool) -> Tuple[str, Optional[str]]:
            log_text, artifact_path = runner.run(files, backend, resolution, skip)
            if artifact_path is None:
                return log_text, None
            return log_text, str(artifact_path)

        run_button.click(
            fn=_execute,
            inputs=[uploads, method, max_resolution, skip_colmap],
            outputs=[logs, artifact],
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.queue(concurrency_count=1).launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))


if __name__ == "__main__":
    main()
