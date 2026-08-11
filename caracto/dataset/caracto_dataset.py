"""General-purpose loader for the published CaRaCTO dataset.

This is the entry point for anyone consuming the dataset outside this repo's own
calibration/reconstruction pipeline (clean JSON/JPEG/NPZ layout, see the dataset
card); `CalibrationSetup` builds on top of it for that pipeline's specific
(legacy-shaped) needs.

Three ways to read a position, depending on what the caller wants:
- `load(key)`: everything as published, unaligned (full camera burst, full radar
  detection list).
- `load_aligned(key)`: camera frames and radar detections trimmed to the
  overlapping window (see `caracto.dataset.alignment`), for callers that want to
  treat frame i of both modalities as roughly the same moment. Raises if the
  position's counts differ too much to align this way.
- `load_single(key)`: one representative measurement (a single camera frame + the
  detection-averaged radar range/azimuth), matching what the calibration pipeline
  has always consumed.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import huggingface_hub
import numpy as np
import numpy.typing as npt

from caracto.dataset.file_readers import read_image_file, read_npz_file


@dataclass
class RadarDetections:
    """A position's radar detection arrays, all index-aligned to each other."""

    range_m: npt.NDArray[np.float32]
    velocity_mps: npt.NDArray[np.float32]
    magnitude: npt.NDArray[np.float32]
    angle_rad: npt.NDArray[np.float32]
    noise: npt.NDArray[np.float32]
    amplitude_re: npt.NDArray[np.float32]
    amplitude_im: npt.NDArray[np.float32]

    def __len__(self) -> int:
        """Return the number of detections."""
        return len(self.range_m)

    def subset(self, index_range: range) -> "RadarDetections":
        """Return a copy with all detection arrays sliced to index_range."""
        s = slice(index_range.start, index_range.stop)
        return replace(
            self,
            range_m=self.range_m[s],
            velocity_mps=self.velocity_mps[s],
            magnitude=self.magnitude[s],
            angle_rad=self.angle_rad[s],
            noise=self.noise[s],
            amplitude_re=self.amplitude_re[s],
            amplitude_im=self.amplitude_im[s],
        )


@dataclass
class PositionSample:
    """One position's data as published: full camera burst + full radar detections."""

    key: str
    annotated: bool
    valid_for_reconstruction: bool
    annotation: dict | None
    ground_truth: dict
    radar: RadarDetections
    camera_frame_paths: list[Path]
    frame_timestamps_ms: list[int]


@dataclass
class SingleMeasurement:
    """One representative frame + averaged radar range/azimuth for a position."""

    key: str
    camera_frame: npt.NDArray[np.uint8]
    camera_frame_index: int
    radar_range_m: float
    radar_azimuth_rad: float
    annotation: dict | None
    ground_truth: dict


class CaractoDataset:
    """Reads the published CaRaCTO-3D dataset from a local path or the Hub."""

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        *,
        repo_id: str = "dfki-av/CaRaCTO-3D",
        revision: str | None = None,
        token: str | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        """Load positions.json from dataset_path, downloading repo_id if unset."""
        if dataset_path is None:
            dataset_path = huggingface_hub.snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                token=token,
                cache_dir=cache_dir,
            )

        self.dataset_path = Path(dataset_path)
        with (self.dataset_path / "positions.json").open() as f:
            self.positions = json.load(f)

    def keys(self) -> list[str]:
        """Return every position's key (e.g. "Position_01")."""
        return list(self.positions.keys())

    def _position_dir(self, key: str) -> Path:
        return self.dataset_path / key

    def load_annotation(self, key: str) -> dict | None:
        """Return key's manual target annotation, or None if unannotated."""
        with (self._position_dir(key) / "annotation.json").open() as f:
            annotation = json.load(f)
        return annotation if annotation["annotated"] else None

    def load_ground_truth(self, key: str) -> dict:
        """Return key's transform-corrected OptiTrack target marker positions."""
        with (self._position_dir(key) / "ground_truth.json").open() as f:
            return json.load(f)

    def load_radar_detections(self, key: str) -> RadarDetections:
        """Return key's full radar detection list."""
        data = read_npz_file(self._position_dir(key) / "radar.npz")
        return RadarDetections(**data)

    def camera_frame_paths(self, key: str) -> list[Path]:
        """Return the paths of key's full camera burst, in order."""
        n = self.positions[key]["num_camera_frames"]
        camera_dir = self._position_dir(key) / "camera"
        return [camera_dir / f"frame_{i:02d}.jpg" for i in range(n)]

    def load_camera_frame(self, key: str, index: int = 0) -> npt.NDArray[np.uint8]:
        """Return one decoded camera frame from key's burst."""
        frame_path = self._position_dir(key) / "camera" / f"frame_{index:02d}.jpg"
        return read_image_file(frame_path)

    def frame_timestamps_ms(self, key: str) -> list[int]:
        """Return key's camera frame timestamps in milliseconds."""
        camera_dir = self._position_dir(key) / "camera"
        with (camera_dir / "frame_timestamps_ms.json").open() as f:
            return json.load(f)

    def load(self, key: str) -> PositionSample:
        """Return key's full PositionSample, unaligned."""
        meta = self.positions[key]
        return PositionSample(
            key=key,
            annotated=meta["annotated"],
            valid_for_reconstruction=meta["valid_for_reconstruction"],
            annotation=self.load_annotation(key),
            ground_truth=self.load_ground_truth(key),
            radar=self.load_radar_detections(key),
            camera_frame_paths=self.camera_frame_paths(key),
            frame_timestamps_ms=self.frame_timestamps_ms(key),
        )

    def load_aligned(self, key: str) -> PositionSample:
        """Return key's camera frames and radar detections, trimmed to their overlap.

        Both modalities are assumed to run at a constant frame rate and start/stop
        within a few cycles of each other; trimmed to their common overlapping
        index range (see `caracto.dataset.alignment.compute_frame_alignment`).
        Raises if this position's counts differ too much to attribute to a simple
        start/stop offset (that's a real detection-count anomaly, not a timing
        offset — check `positions.json`'s `frame_alignment.note` for that
        position).
        """
        alignment = self.positions[key]["frame_alignment"]
        if not alignment["aligned"]:
            msg = f"{key} has no well-defined frame alignment: {alignment['note']}"
            raise ValueError(msg)

        sample = self.load(key)
        cam_lo, cam_hi = alignment["camera_frame_range"]
        rad_lo, rad_hi = alignment["radar_detection_range"]
        sample.camera_frame_paths = sample.camera_frame_paths[cam_lo:cam_hi]
        sample.frame_timestamps_ms = sample.frame_timestamps_ms[cam_lo:cam_hi]
        sample.radar = sample.radar.subset(range(rad_lo, rad_hi))
        return sample

    def load_single(self, key: str, camera_frame_index: int = 0) -> SingleMeasurement:
        """Return one representative frame + averaged radar range/azimuth for key.

        Matches what the calibration pipeline has always consumed
        (`CalibrationSetup.get_radar_measurements`'s averaging, including its
        sign convention on azimuth).
        """
        radar = self.load_radar_detections(key)
        return SingleMeasurement(
            key=key,
            camera_frame=self.load_camera_frame(key, camera_frame_index),
            camera_frame_index=camera_frame_index,
            radar_range_m=float(np.mean(radar.range_m)),
            radar_azimuth_rad=float(-np.mean(radar.angle_rad)),
            annotation=self.load_annotation(key),
            ground_truth=self.load_ground_truth(key),
        )

    def load_camera_intrinsics(self) -> dict:
        """Return the camera matrix and distortion coefficients."""
        with (self.dataset_path / "calibration" / "camera_intrinsics.json").open() as f:
            return json.load(f)

    def load_axis_convention(self) -> dict:
        """Return the published coordinate frame's axis convention."""
        with (self.dataset_path / "calibration" / "axis_convention.json").open() as f:
            return json.load(f)

    def load_radar_rig_markers(self) -> dict:
        """Return the fixed camera+radar rig's 4 mocap marker positions."""
        with (self.dataset_path / "calibration" / "radar_rig_markers.json").open() as f:
            return json.load(f)
