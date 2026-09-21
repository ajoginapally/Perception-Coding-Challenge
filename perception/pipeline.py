"""Connect frame loading, measurement, coordinate transforms, and rendering."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from .data import frame_id, read_dataset, sample_xyz, write_csv
from .geometry import camera_position, reference_angle, object_position, local_xy
from .geometry import fill_short_gaps, smooth_segments
from .tracking import ObjectTracker, SignalColorTracker, load_seeds
from .render import render_outputs


def load_yaw(path, frame_ids):
    """Yaw is counterclockwise/left-positive in radians relative to frame zero."""
    if path is None:
        return np.zeros(len(frame_ids))
    values = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"frame_id", "yaw_rad"} <= set(reader.fieldnames or []):
            raise ValueError("Yaw CSV needs frame_id,yaw_rad")
        for row in reader:
            fid, yaw = frame_id(row["frame_id"]), float(row["yaw_rad"])
            if fid in values or not np.isfinite(yaw):
                raise ValueError(f"Duplicate or invalid yaw for frame {fid}")
            values[fid] = yaw
    missing = set(frame_ids) - values.keys()
    if missing:
        raise ValueError(f"Missing yaw for frames {sorted(missing)[:10]}")
    yaw = np.array([values[fid] for fid in frame_ids])
    return np.unwrap(yaw) - yaw[0]


def run(args):
    frames = read_dataset(args.dataset, args.allow_missing_depth, require_boxes=args.light_box is None)
    ids = np.array([frame[0] for frame in frames])
    times = (ids - ids[0]) / args.fps
    yaw = load_yaw(args.yaw_csv, ids)
    seeds = load_seeds(args.objects_csv)
    if seeds.keys() - set(ids):
        raise ValueError("Object seeds refer to frames absent from the dataset")
    tracker = ObjectTracker()
    light_tracker = SignalColorTracker() if args.light_box is not None else None
    tracked_boxes = []
    used_boxes = []
    light_rows, object_rows, light_points = [], [], []
    dimensions = None
    for index, (fid, image_path, xyz_path, box) in enumerate(frames):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read RGB frame: {image_path}")
        points = None
        if xyz_path is not None:
            with np.load(xyz_path, allow_pickle=False) as data:
                key = "points" if "points" in data else "xyz"
                if key not in data:
                    raise ValueError(f"Missing points/xyz array: {xyz_path}")
                points = data[key]
            if points.ndim != 3 or points.shape[:2] != image.shape[:2] or points.shape[2] not in (3,4):
                raise ValueError(f"XYZ/RGB shape mismatch at frame {fid}: {points.shape}, {image.shape}")
            points = points[:,:,:3].copy()  # fourth channel is not a coordinate
            if not np.issubdtype(points.dtype, np.floating):
                raise ValueError(f"XYZ array must be floating point in {xyz_path}")
            if args.camera_y == "left":
                points[:,:,1] *= -1  # normalize to internal forward/right/up convention
        if light_tracker is not None:
            box = light_tracker.update(image, args.light_box if index == 0 else None)
            if box is not None:
                tracked_boxes.append(dict(frame_id=int(fid), x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]))
        used_boxes.append(box)
        if dimensions is not None and image.shape != dimensions:
            raise ValueError("RGB dimensions changed between frames")
        dimensions = image.shape
        measurement = sample_xyz(points, box, radius=args.patch_radius,
                                 min_samples=args.min_samples) if points is not None else dict(xyz=np.full(3,np.nan), count=0, spread=np.nan, status="missing_depth_file")
        xyz = measurement["xyz"]
        light_points.append(xyz)
        light_rows.append(dict(frame_id=int(fid), time_s=float(times[index]),
                               light_X=float(xyz[0]), light_Y=float(xyz[1]), light_Z=float(xyz[2]),
                               valid_samples=measurement["count"], forward_depth_mad=measurement["spread"],
                               measurement_status=measurement["status"]))
        if seeds:
            for track in tracker.update(image, seeds.get(fid, [])):
                depth = sample_xyz(points, track.box, min_samples=args.min_samples, interior=True) if track.status != "lost" and points is not None else {
                    "xyz": np.full(3,np.nan), "count": 0, "spread": np.nan, "status": "track_lost" if track.status == "lost" else "missing_depth_file"}
                x, y, z = depth["xyz"]
                ex, ey = local_xy(depth["xyz"])
                object_rows.append(dict(frame_id=int(fid), time_s=float(times[index]),
                    object_id=track.object_id, label=track.label,
                    tracking_status=track.status, tracking_inliers=track.inliers,
                    x_min=float(track.box[0]), y_min=float(track.box[1]),
                    x_max=float(track.box[2]), y_max=float(track.box[3]),
                    camera_X=float(x), camera_Y=float(y), camera_Z=float(z),
                    ego_x=float(ex), ego_y=float(ey), world_x=np.nan, world_y=np.nan,
                    valid_samples=depth["count"], forward_depth_mad=depth["spread"], depth_status=depth["status"]))
        if index % 25 == 0 or index == len(frames)-1:
            print(f"Measured {index+1}/{len(frames)} frames", flush=True)

    light_points = np.asarray(light_points)
    
    if not np.isfinite(light_points[0]).all():
        raise ValueError("First frame has no reliable light depth; inspect its box/patch before defining the world frame")
    base_angle = reference_angle(light_points[0])
    angles = base_angle + yaw
    raw = np.array([camera_position(p,a) for p,a in zip(light_points,angles)])
    accepted = raw.copy()
    rejected = np.zeros(len(frames), dtype=bool)
   
   
    if args.max_speed is not None:
        for k in range(1, len(frames)-1):
            a,b,c = raw[k-1:k+2]
            if not np.isfinite([a,b,c]).all():
                continue
            before = np.linalg.norm(b-a)/(times[k]-times[k-1])
            after = np.linalg.norm(c-b)/(times[k+1]-times[k])
            bridge = np.linalg.norm(c-a)/(times[k+1]-times[k-1])
            if min(before,after) > args.max_speed and bridge <= args.max_speed:
                accepted[k] = np.nan
                rejected[k] = True
    
    filled, interpolated = fill_short_gaps(accepted, args.max_gap, times)
    positions = smooth_segments(filled, args.smooth_window)
    
    
    positions[0] = raw[0]
    for k, row in enumerate(light_rows):
        row.update(yaw_change_rad=float(yaw[k]), raw_x=float(raw[k,0]), raw_y=float(raw[k,1]),
                   x_m=float(positions[k,0]), y_m=float(positions[k,1]),
                   rejected_spike=bool(rejected[k]), interpolated=bool(interpolated[k]))
    index_by_id = {int(fid): k for k,fid in enumerate(ids)}
    
    for row in object_rows:
        k = index_by_id[row["frame_id"]]
        world = object_position([row["camera_X"],row["camera_Y"],row["camera_Z"]], positions[k], angles[k])
        row["world_x"], row["world_y"] = map(float,world)
        
        
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "trajectory.csv", light_rows)
    if tracked_boxes:
        write_csv(output / "tracked_light_boxes.csv", tracked_boxes)
    if object_rows:
        write_csv(output / "objects.csv", object_rows)
    summary = dict(
        frames=len(frames), fps=args.fps, fps_note=args.fps_note,
        camera_y_input=args.camera_y, depth_files=sum(f[2] is not None for f in frames),
        light_box_source="manual initialization + red/green component tracking" if light_tracker else "provided CSV", duration_between_first_last_s=float(times[-1]),
        valid_light_measurements=int(np.isfinite(light_points).all(axis=1).sum()),
        rejected_spikes=int(rejected.sum()), interpolated_positions=int(interpolated.sum()),
        missing_positions=int((~np.isfinite(positions).all(axis=1)).sum()),
        heading_mode="provided yaw" if args.yaw_csv else "constant heading",
        origin="ground directly below traffic light",
        world_axes="initial car-to-light direction, left, up",
        reference_point="camera ground projection; no unknown mounting offset applied",
        assumptions=["level camera axes", "locally planar ground", "consistent physical point sampled on light"],
        smoothing_window=args.smooth_window, patch_radius=args.patch_radius,
        min_samples=args.min_samples, max_gap=args.max_gap, max_speed_m_s=args.max_speed,
        objects_frame=args.object_frame, object_ids=sorted({row["object_id"] for row in object_rows}))
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    render_outputs(output, frames, times, raw, positions, angles, object_rows, args.fps,
                   used_boxes, make_video=not args.no_video,
                   object_frame=args.object_frame, mode=summary["heading_mode"])
    print(f"Saved results to {output.resolve()}")
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True, help="Folder containing rgb/, xyz/, bboxes_light.csv")
    result.add_argument("--fps", type=float, required=True, help="Source frame rate; frame ID differences define elapsed time")
    result.add_argument("--output", type=Path, default=Path("outputs"))
    result.add_argument("--patch-radius", type=int, default=2, help="Light patch radius in pixels (2 means 5x5)")
    result.add_argument("--min-samples", type=int, default=3)
    result.add_argument("--max-gap", type=int, default=0, help="Maximum missing observations to interpolate; default leaves gaps")
    result.add_argument("--smooth-window", type=int, default=1, help="Odd centered mean window; 1 disables smoothing")
    result.add_argument("--max-speed", type=float, help="Optional isolated-spike threshold in meters/second")
    result.add_argument("--yaw-csv", type=Path, help="Optional frame_id,yaw_rad; left-positive camera yaw")
    result.add_argument("--objects-csv", type=Path, help="Optional object seed/reseed boxes for Part B")
    result.add_argument("--object-frame", choices=["ego", "world"], default="ego")
    result.add_argument("--camera-y", choices=["left", "right"], default="right")
    result.add_argument("--allow-missing-depth", action="store_true", help="Retain RGB frames without depth as missing measurements")
    result.add_argument("--light-box", nargs=4, type=float, metavar=("X1","Y1","X2","Y2"), help="Initialize RGB light tracking instead of requiring a box CSV")
    result.add_argument("--fps-note", default="user-supplied source frame rate")
    result.add_argument("--no-video", action="store_true", help="Skip video for quick inspection; submission requires MP4")
    return result


def main():
    command = parser()
    args = command.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        command.error("--fps must be finite and positive")
    if args.patch_radius < 0 or args.min_samples < 1 or args.max_gap < 0:
        command.error("patch radius/max gap must be nonnegative and min samples positive")
    if args.smooth_window < 1 or args.smooth_window % 2 != 1:
        command.error("--smooth-window must be a positive odd integer")
    if args.max_speed is not None and (not np.isfinite(args.max_speed) or args.max_speed <= 0):
        command.error("--max-speed must be finite and positive")
    if args.light_box is not None:
        box = np.asarray(args.light_box)
        if not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            command.error("--light-box needs finite, increasing corners")
    try:
        run(args)
    except (OSError, ValueError, KeyError) as error:
        command.exit(1, f"Error: {error}\n")
