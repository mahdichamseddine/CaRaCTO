"""Index-based alignment between a position's camera burst and radar detections.

Camera and radar are captured by independently-clocked acquisition loops that
start/stop within a few cycles of each other, so their per-position frame/detection
counts are usually equal or off by only a couple of frames. There are no reliable
per-detection timestamps on the radar side, so alignment assumes a constant frame
rate per modality (not wall-clock timestamps) and trims the longer sequence's extra
frames symmetrically from the start/end to line up index-for-index.

A few positions have far larger count mismatches (radar detection dropouts/extra
noise detections rather than a start/stop timing offset) — those are intentionally
left unaligned rather than force-trimmed, since a large trim would silently discard
most of the sequence.
"""

MAX_SYMMETRIC_TRIM = 4


def compute_frame_alignment(
    num_camera_frames: int,
    num_radar_detections: int,
    max_symmetric_trim: int = MAX_SYMMETRIC_TRIM,
) -> dict:
    """Return the trimmed, index-aligned camera/radar ranges for one position."""
    diff = num_camera_frames - num_radar_detections

    if abs(diff) > max_symmetric_trim:
        return {
            "aligned": False,
            "camera_frame_range": None,
            "radar_detection_range": None,
            "note": (
                f"camera ({num_camera_frames}) and radar ({num_radar_detections}) "
                f"counts differ by {diff}, too large to attribute to acquisition "
                "start/stop timing alone (likely radar detection dropouts/extra "
                "detections rather than a simple offset); left unaligned."
            ),
        }

    n = min(num_camera_frames, num_radar_detections)
    cam_extra = num_camera_frames - n
    rad_extra = num_radar_detections - n
    cam_start = cam_extra // 2
    rad_start = rad_extra // 2

    return {
        "aligned": True,
        "camera_frame_range": [cam_start, cam_start + n],
        "radar_detection_range": [rad_start, rad_start + n],
        "note": None,
    }
