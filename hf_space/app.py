from __future__ import annotations

import datetime as dt
import io
import json
import os
import shlex
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
    runner: Callable[[Path, Path, Optional[Dict[str, Path]], int], Tuple[Path, List[str], Optional[Path]]]


@dataclass
class RoomInputs:
    """Container for the multiple user-provided signals."""

    room_images: Iterable[Any]
    floor_plan: Optional[Any]
    depth_maps: Iterable[Any]
    camera_parameters: Optional[Any]
    annotations: str


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
        inputs: RoomInputs,
        method: str,
        max_resolution: int,
        skip_colmap: bool,
    ) -> Tuple[str, Optional[Path], Optional[Path]]:
        logs: List[str] = []
        timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        workspace = self.output_root / f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
        dataset_root = workspace / "dataset"
        images_dir = dataset_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        logs.append(f"Workspace initialized at {workspace}")

        try:
            ingest_count = self._ingest_room_inputs(inputs, dataset_root, max_resolution)
        except Exception as exc:  # noqa: BLE001 - top-level guard for user feedback
            logs.append(f"[ERROR] Failed to ingest inputs: {exc}")
            return "\n".join(logs), None, None

        if ingest_count == 0:
            logs.append("[ERROR] No images detected in upload. Provide JPG/PNG files or a ZIP archive.")
            return "\n".join(logs), None, None

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
                return "\n".join(logs), None, None
            except RuntimeError as exc:
                logs.append(str(exc))
                return "\n".join(logs), None, None

        backend = self.backends.get(method)
        if not backend:
            logs.append(f"[ERROR] Unknown backend '{method}'. Available options: {', '.join(self.available_methods())}")
            return "\n".join(logs), None, None

        try:
            artifact_path, backend_logs, preview_path = backend.runner(
                workspace, dataset_root, colmap_outputs, max_resolution
            )
            logs.extend(backend_logs)
        except Exception as exc:  # noqa: BLE001 - propagate details to UI
            logs.append(f"[ERROR] Backend '{method}' failed: {exc}")
            return "\n".join(logs), None, None

        logs.append(f"Artifacts packaged at {artifact_path}")
        if preview_path:
            logs.append(f"Preview geometry located at {preview_path}")
        return "\n".join(logs), artifact_path, preview_path

    # ------------------------------------------------------------------
    # Backend registration
    # ------------------------------------------------------------------
    def register_backend(self, backend: Backend) -> None:
        self.backends[backend.name] = backend

    def _register_default_backends(self) -> None:
        self.register_backend(
            Backend(
                name="COLMAP Room Point Cloud",
                description=(
                    "Runs the full COLMAP indoor pipeline (feature extraction, dense stereo, Poisson meshing) and "
                    "packages the sparse model, fused point cloud, and mesh preview in a ZIP archive."
                ),
                runner=self._run_colmap_pointcloud,
            )
        )
        self.register_backend(
            Backend(
                name="VGGT Depth Fusion (optional)",
                description=(
                    "After COLMAP pose estimation, expects a VGGT PointMap depth fusion command to be provided via "
                    "the `VGGT_ROOM_COMMAND` environment variable. The command receives image/depth paths through "
                    "environment variables and should export a mesh; falls back to COLMAP results when absent."
                ),
                runner=self._run_vggt_depth_fusion,
            )
        )

    # ------------------------------------------------------------------
    # Input ingestion helpers
    # ------------------------------------------------------------------
    def _ingest_room_inputs(self, inputs: RoomInputs, dataset_root: Path, max_resolution: int) -> int:
        images_dir = dataset_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata: List[Dict[str, object]] = []
        count = 0

        def _resolve_path(item: Any) -> Optional[Path]:
            potential = Path(getattr(item, "name", getattr(item, "path", "")))
            if potential.exists():
                return potential
            if hasattr(item, "path"):
                alt = Path(item.path)
                if alt.exists():
                    return alt
            return None

        for item in inputs.room_images or []:
            if not item:
                continue
            src_path = _resolve_path(item)
            if not src_path:
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
            if inputs.annotations:
                dataset_meta["notes"] = inputs.annotations
            meta_path = dataset_root / "metadata.json"
            meta_path.write_text(json.dumps(dataset_meta, indent=2))

        if inputs.floor_plan:
            floor_plan_dir = dataset_root / "floor_plan"
            floor_plan_dir.mkdir(parents=True, exist_ok=True)
            src_path = _resolve_path(inputs.floor_plan)
            if src_path and zipfile.is_zipfile(src_path):
                with zipfile.ZipFile(src_path, "r") as archive:
                    archive.extractall(floor_plan_dir)
            elif src_path:
                shutil.copy(src_path, floor_plan_dir / src_path.name)

        if inputs.depth_maps:
            depth_dir = dataset_root / "depth_maps"
            depth_dir.mkdir(parents=True, exist_ok=True)
            for item in inputs.depth_maps:
                if not item:
                    continue
                src_path = _resolve_path(item)
                if not src_path:
                    continue
                if zipfile.is_zipfile(src_path):
                    with zipfile.ZipFile(src_path, "r") as archive:
                        archive.extractall(depth_dir)
                else:
                    shutil.copy(src_path, depth_dir / src_path.name)

        if inputs.camera_parameters:
            camera_dir = dataset_root / "cameras"
            camera_dir.mkdir(parents=True, exist_ok=True)
            src_path = _resolve_path(inputs.camera_parameters)
            if src_path:
                shutil.copy(src_path, camera_dir / src_path.name)

        if inputs.annotations:
            notes_path = dataset_root / "user_notes.txt"
            notes_path.write_text(inputs.annotations)

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

    @staticmethod
    def _zip_artifacts(destination: Path, paths: Iterable[Optional[Path]]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            base = destination.parent
            for maybe_path in paths:
                if not maybe_path:
                    continue
                path = Path(maybe_path)
                if path.is_file():
                    arcname = ReconstructionRunner._relative_to_base(path, base)
                    archive.write(path, arcname=arcname)
                else:
                    for file_path in path.rglob("*"):
                        if file_path.is_file():
                            arcname = ReconstructionRunner._relative_to_base(file_path, base)
                            archive.write(file_path, arcname=arcname)

    @staticmethod
    def _relative_to_base(path: Path, base: Path) -> str:
        try:
            return str(path.relative_to(base))
        except ValueError:
            return path.name

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
            (
                "Patch match stereo",
                [
                    "colmap",
                    "patch_match_stereo",
                    "--workspace_path",
                    str(dense_dir),
                    "--workspace_format",
                    "COLMAP",
                    "--PatchMatchStereo.geom_consistency",
                    "1",
                ],
            ),
            (
                "Stereo fusion",
                [
                    "colmap",
                    "stereo_fusion",
                    "--workspace_path",
                    str(dense_dir),
                    "--workspace_format",
                    "COLMAP",
                    "--input_type",
                    "geometric",
                    "--output_path",
                    str(dense_dir / "fused.ply"),
                ],
            ),
            (
                "Poisson mesher",
                [
                    "colmap",
                    "poisson_mesher",
                    "--input_path",
                    str(dense_dir / "fused.ply"),
                    "--output_path",
                    str(dense_dir / "meshed-poisson.ply"),
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
    def _run_colmap_pointcloud(
        self,
        workspace: Path,
        dataset_root: Path,
        colmap_outputs: Optional[Dict[str, Path]],
        max_resolution: int,
    ) -> Tuple[Path, List[str], Optional[Path]]:
        if not colmap_outputs:
            raise RuntimeError("COLMAP must run before exporting the room point cloud.")

        logs: List[str] = ["Packaging COLMAP dense reconstruction…"]
        dense_dir = colmap_outputs.get("dense")
        sparse_dir = colmap_outputs.get("sparse")
        database_path = colmap_outputs.get("database")

        preview_path: Optional[Path] = None
        fused_ply = dense_dir / "fused.ply" if dense_dir else None
        poisson_ply = dense_dir / "meshed-poisson.ply" if dense_dir else None
        if poisson_ply and poisson_ply.exists():
            preview_path = poisson_ply
            logs.append("Using Poisson mesh as preview artifact.")
        elif fused_ply and fused_ply.exists():
            preview_path = fused_ply
            logs.append("Poisson mesh missing, falling back to fused point cloud preview.")
        else:
            logs.append("Dense outputs missing; preview will not be displayed.")

        artifact_path = workspace / "colmap_room_outputs.zip"
        include_paths = [p for p in [dataset_root, sparse_dir, dense_dir, database_path] if p is not None]
        self._zip_artifacts(artifact_path, include_paths)
        logs.append("COLMAP artifacts archived.")
        return artifact_path, logs, preview_path

    def _run_vggt_depth_fusion(
        self,
        workspace: Path,
        dataset_root: Path,
        colmap_outputs: Optional[Dict[str, Path]],
        max_resolution: int,
    ) -> Tuple[Path, List[str], Optional[Path]]:
        artifact_path, fallback_logs, preview = self._run_colmap_pointcloud(
            workspace, dataset_root, colmap_outputs, max_resolution
        )
        logs: List[str] = []
        logs.extend(fallback_logs)

        command_text = os.environ.get("VGGT_ROOM_COMMAND")
        if not command_text:
            logs.append(
                "VGGT_ROOM_COMMAND environment variable not set. Skipping VGGT fusion and returning COLMAP outputs."
            )
            return artifact_path, logs, preview

        logs.append("VGGT command detected; attempting depth-guided fusion…")
        try:
            command = shlex.split(command_text)
        except ValueError as exc:
            logs.append(f"[WARNING] Failed to parse VGGT_ROOM_COMMAND: {exc}. Returning COLMAP artifacts only.")
            return artifact_path, logs, preview

        vggt_output_dir = workspace / "vggt_fusion"
        vggt_output_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "VGGT_INPUT_IMAGES": str(dataset_root / "images"),
                "VGGT_INPUT_DEPTH": str((dataset_root / "depth_maps")),
                "VGGT_INPUT_CAMERAS": str((dataset_root / "cameras")),
                "VGGT_WORKSPACE": str(vggt_output_dir),
                "VGGT_MAX_RESOLUTION": str(max_resolution),
            }
        )

        logs.append(f"\n$ {' '.join(command)}")
        code, output = _run_command(command, cwd=vggt_output_dir, env=env)
        logs.append(output)
        if code != 0:
            logs.append(
                "VGGT command exited with a non-zero status. Check the logs above; returning COLMAP reconstruction instead."
            )
            return artifact_path, logs, preview

        mesh_candidates = sorted(vggt_output_dir.rglob("*.ply")) + sorted(vggt_output_dir.rglob("*.obj"))
        if mesh_candidates:
            preview = mesh_candidates[-1]
            logs.append(f"VGGT fusion produced preview mesh: {preview}")
        else:
            logs.append("VGGT fusion completed but no mesh file was detected; preview will remain the COLMAP output.")

        # Rebuild archive with VGGT assets appended alongside the COLMAP bundle.
        include_paths = [
            dataset_root,
            colmap_outputs.get("sparse") if colmap_outputs else None,
            colmap_outputs.get("dense") if colmap_outputs else None,
            colmap_outputs.get("database") if colmap_outputs else None,
            vggt_output_dir,
        ]
        self._zip_artifacts(artifact_path, include_paths)
        logs.append("VGGT fusion artifacts archived alongside COLMAP outputs.")
        return artifact_path, logs, preview


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

    with gr.Blocks(title="Indoor Room ➜ 3D Model Studio") as demo:
        gr.Markdown(
            textwrap.dedent(
                """
                # Indoor Room ➜ 3D Model Studio

                Provide photographs of a single room (JPEG/PNG or a ZIP archive). The app runs **COLMAP** to recover camera
                poses and dense geometry, optionally calling out to a **VGGT PointMap** command for depth fusion when available.
                Auxiliary signals like floor plans, pre-computed depth maps, or camera intrinsics help stabilize reconstructions.
                """
            )
        )

        with gr.Row():
            room_uploads = gr.Files(
                label="Room photos or ZIP archive",
                file_types=["image", ".zip"],
                file_count="multiple",
                info="Capture the room from multiple angles with consistent exposure.",
            )
            method = gr.Dropdown(
                choices=runner.available_methods(),
                value=runner.available_methods()[0] if runner.available_methods() else "",
                label="Reconstruction backend",
            )

        with gr.Accordion("Optional indoor priors", open=False):
            with gr.Row():
                floor_plan = gr.File(
                    label="Floor plan (image or ZIP)",
                    file_types=["image", ".zip", ".pdf"],
                    file_count="single",
                    info="Helps downstream mesh generation align with layout when using custom VGGT commands.",
                )
                depth_maps = gr.Files(
                    label="Depth maps (images or ZIP)",
                    file_types=["image", ".zip"],
                    file_count="multiple",
                    info="Provide precomputed depth estimates to reuse with VGGT fusion.",
                )
            with gr.Row():
                camera_params = gr.File(
                    label="Camera intrinsics (JSON/TXT)",
                    file_types=[".json", ".txt", ".csv"],
                    file_count="single",
                )
                annotations = gr.Textbox(
                    label="Notes for the run",
                    placeholder="Room type, lighting conditions, or alignment hints…",
                    lines=3,
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
        preview = gr.Model3D(label="Preview mesh/point cloud", clear_color=[0.1, 0.1, 0.1, 1.0])

        def _execute(
            files: List[Any],
            floor: Optional[Any],
            depths: List[Any],
            cameras: Optional[Any],
            notes: str,
            backend: str,
            resolution: int,
            skip: bool,
        ) -> Tuple[str, Optional[str], Optional[str]]:
            structured_inputs = RoomInputs(
                room_images=files or [],
                floor_plan=floor,
                depth_maps=depths or [],
                camera_parameters=cameras,
                annotations=notes or "",
            )
            log_text, artifact_path, preview_path = runner.run(structured_inputs, backend, resolution, skip)
            artifact_value = str(artifact_path) if artifact_path else None
            preview_value = str(preview_path) if preview_path else None
            return log_text, artifact_value, preview_value

        run_button.click(
            fn=_execute,
            inputs=[room_uploads, floor_plan, depth_maps, camera_params, annotations, method, max_resolution, skip_colmap],
            outputs=[logs, artifact, preview],
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.queue(concurrency_count=1).launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))


if __name__ == "__main__":
    main()
