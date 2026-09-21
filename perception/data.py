"""Strict frame synchronization and robust, bounded XYZ sampling."""
import csv
import re
from pathlib import Path
import numpy as np


def frame_id(value):
    match = re.fullmatch(r"(?:frame_|left|depth)?(\d+)(?:\.[A-Za-z0-9]+)?", str(value).strip())
    if not match:
        raise ValueError(f"Unrecognized frame ID: {value!r}")
    return int(match.group(1))


def index_files(directory, suffixes):
    result = {}
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() not in suffixes:
            continue
        number = frame_id(path.stem)
        if number in result:
            raise ValueError(f"Duplicate frame {number} in {directory}")
        result[number] = path
    return result


def read_dataset(root, allow_missing_depth=False, require_boxes=True):
    root = Path(root)
    images = index_files(root / "rgb", {".png", ".jpg", ".jpeg"})
    clouds = index_files(root / "xyz", {".npz"})
    if not images or (not allow_missing_depth and images.keys() != clouds.keys()):
        raise ValueError("RGB and XYZ frame IDs must match exactly and be nonempty")
    boxes = {}
    if not (root / "bboxes_light.csv").exists() and not require_boxes:
        return [(n, images[n], clouds.get(n), None) for n in sorted(images)]
    with (root / "bboxes_light.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_id", "x_min", "y_min", "x_max", "y_max"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Bounding-box CSV needs columns {sorted(required)}")
        for row in reader:
            number = frame_id(row["frame_id"])
            if number in boxes:
                raise ValueError(f"Duplicate light box for frame {number}")
            box = np.array([float(row[k]) for k in ("x_min", "y_min", "x_max", "y_max")])
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Invalid light box in frame {number}: {box}")
            boxes[number] = box
    if boxes.keys() - images.keys():
        raise ValueError("Bounding-box CSV contains IDs without RGB/XYZ frames")
    return [(n, images[n], clouds.get(n), boxes.get(n)) for n in sorted(images)]


def sample_xyz(points, box, radius=2, min_samples=3, interior=False):
    """Median of finite, forward-facing samples, with forward-depth MAD gate.

    This resists sparse outliers, not a majority of background pixels. Light
    patches stay inside the box. Object sampling uses the inner half of its box.
    """
    missing = dict(xyz=np.full(3, np.nan), count=0, spread=np.nan, status="missing_box")
    if box is None:
        return missing
    h, w = points.shape[:2]
    x1, y1, x2, y2 = box
    u, v = (x1+x2)/2, (y1+y2)/2
    if not (0 <= u < w and 0 <= v < h):
        return {**missing, "status": "center_outside_image"}
    if interior:
        rx, ry = max(1, (x2-x1)/4), max(1, (y2-y1)/4)
        left, right = int(np.ceil(u-rx)), int(np.floor(u+rx))+1
        top, bottom = int(np.ceil(v-ry)), int(np.floor(v+ry))+1
    else:
        uc, vc = int(np.floor(u)), int(np.floor(v))
        left, right = max(uc-radius, int(np.ceil(x1))), min(uc+radius+1, int(np.ceil(x2)))
        top, bottom = max(vc-radius, int(np.ceil(y1))), min(vc+radius+1, int(np.ceil(y2)))
    samples = points[max(0,top):min(h,bottom), max(0,left):min(w,right)].reshape(-1,3)
    valid = np.isfinite(samples).all(axis=1) & (samples[:,0] > 0)
    samples = samples[valid]
    if len(samples) < min_samples:
        return {**missing, "count": len(samples), "status": "insufficient_depth"}
    median = np.median(samples[:,0])
    mad = np.median(np.abs(samples[:,0]-median))
    samples = samples[np.abs(samples[:,0]-median) <= max(0.05, 3*1.4826*mad)]
    if len(samples) < min_samples:
        return {**missing, "count": len(samples), "status": "insufficient_inliers"}
    return dict(xyz=np.median(samples, axis=0), count=len(samples),
                spread=float(np.median(np.abs(samples[:,0]-np.median(samples[:,0])))), status="ok")


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
