# Implementation walkthrough

The original README is preserved. This guide explains the Python implementation;
it does not claim results on the challenge recording. The dataset is not yet in
this folder, and real-data validation is still pending.

## Environment

This uses the existing uv project, `pyproject.toml`, `uv.lock`, and `.venv`.
Run `uv sync` to reproduce the dependency environment. NumPy handles XYZ arrays,
OpenCV handles images and tracking, Matplotlib draws the plots, and
imageio-ffmpeg supplies an FFmpeg executable for MP4 encoding. No separate
system FFmpeg installation is required by this implementation.

## 1. Frame synchronization — perception/data.py

The reader pairs RGB and XYZ files by numeric frame ID, not filesystem order or
CSV row position. It checks duplicate IDs and requires corresponding image and
cloud files. A missing light box becomes a missing observation. Malformed boxes
or annotations for nonexistent frames produce an error.

Each point cloud must have key `points` and shape `(image_height,image_width,3)`.
Only one cloud is loaded at a time. Frame IDs may skip numbers; their differences
and the user-provided source FPS determine elapsed time. Video export holds the
previous image for skipped source frames. This assumes IDs represent source
frame numbers, not arbitrary labels.

## 2. Robust light depth — perception/data.py

The light measurement comes from a small central patch clipped to the box and
image. NumPy indexing is `[row,column]`, or `[v,u]`. Remove nonfinite XYZ vectors
and nonpositive forward distances. Keep valid zero lateral or vertical values.
Filter forward-depth outliers using a median absolute deviation gate, then take
a coordinate-wise median of the retained XYZ points. Save sample count, forward
depth MAD (meters), and a reason for every missing measurement.

Default patch radius 2 means up to 25 samples. Single-pixel sampling is available
with `--patch-radius 0 --min-samples 1`. More samples are not always better: a
background-dominated patch gives the wrong depth even with robust statistics.
Inspect the box and patch on real images before trusting this estimator.

## 3. Ground-frame geometry — perception/geometry.py

The stated input axes are forward/right/up, a left-handed convention. Convert
horizontal measurements to forward/left as `q = [X,-Y]`. For initial light
vector q0, set `base_angle = -atan2(q0_y,q0_x)`. This rotates the initial
car-to-light line onto world +X.

With `R(angle)` the usual planar rotation, camera position is `-R @ q` because
the light's world horizontal position is zero. Initial camera XY is therefore
`[-norm(q0),0]`. Output tracks the camera's ground projection; no unprovided
camera-to-vehicle offset is invented.

The default assumes constant heading and level axes. One light cannot recover
both unknown translation and yaw. If independently measured camera yaw is
available, `--yaw-csv` accepts `frame_id,yaw_rad`, left-positive radians, with
one row for every dataset frame. Yaw is unwrapped and its initial value removed.
The angle becomes `base_angle + yaw_change`. This is an input interface, not an
implemented visual-odometry estimator. Tilt compensation is also not implemented.

A valid initial light observation is required. The program refuses to silently
use a later reference frame, since that changes the specified world definition.

## 4. Temporal processing — perception/pipeline.py

Raw positions are retained in the output CSV. By default, there is no temporal
smoothing, interpolation, or speed rejection. Tune these only after inspection:

- `--max-speed M` rejects an isolated middle point only if both adjacent speeds
  exceed M meters/second while its two neighbors have a plausible average speed.
  It is not a general motion estimator or a replacement for depth validation.
- `--max-gap N` fills at most N consecutive missing observations between valid
  endpoints, using time-weighted linear interpolation. It never extrapolates.
- `--smooth-window N` uses an odd centered moving-average window within each
  finite segment. It never averages across a remaining gap. The first position
  remains exactly anchored to the chosen world reference.

The CSV records raw XYZ, raw world XY, output XY, rejection and interpolation
flags. The JSON summary records assumptions and processing parameters.

## 5. Object tracking — perception/tracking.py

Part B uses a seed CSV, supplied after inspecting RGB frames:

```csv
frame_id,object_id,label,x_min,y_min,x_max,y_max
```

Give each object a unique stable ID such as `barrel_1` or `cart_1`. Specify its
initial bounding box in original-resolution pixels. Additional rows for that
same ID on later frames reinitialize it after drift or occlusion. Coordinates
must lie inside the image. These are manual initializations, not an automatic
object detector.

For each active region, detect Shi–Tomasi corners. Pyramidal Lucas–Kanade follows
those patches to the next image. Track them backwards too: a match is rejected
if it fails the forward/backward consistency check. Fit a similarity transform
(translation, rotation, and uniform scale) with RANSAC, requiring at least six
inliers and a reasonable scale change. Transform the box and refresh features
within it for the next frame.

If matching fails, mark the track lost. Do not invent its current position.
Reinitialization requires a later seed. Textureless barrels, background-heavy
boxes, axis-aligned box growth during rotation, and occlusion can cause drift;
inspect the RGB overlays and reseed when necessary. A foreground mask or a
validated detector would improve this baseline later.

Object XYZ uses the central half of each current box with the same validity and
robust-depth rules. This estimates a visible-region position, not a physical
3D object centroid. An empty/background region can still mislead it.

Ego BEV uses `[X,-Y]` directly. World BEV uses `camera_xy + R @ object_local_xy`.
The latter inherits the ego estimate's heading and depth errors. Static barrels
should stay approximately fixed in world coordinates and can diagnose drift.

## 6. Output — perception/render.py

`trajectory.png` and `trajectory.mp4` always use the traffic-light world frame.
Axes are in meters, equally scaled, and fixed through the animation. Missing
positions break the path and hide the current-position marker.

With object seeds, `objects_bev.mp4` adds synchronized RGB tracking overlays and
an object BEV. Select `--object-frame ego` (default) or `world`. Lost tracks are
not plotted. Tracks with valid image locations but invalid depth retain their
RGB annotation with a depth-status label and have no BEV marker.

The required outputs are accompanied by `trajectory.csv`, `run_summary.json`,
and, when Part B is enabled, `objects.csv`. Each rerun replaces same-named output
files, so use a different `--output` directory to compare configurations.

## Running after the code-review stage

Replace DATASET_PATH and SOURCE_FPS with the real dataset folder and frame rate:

```bash
uv run main.py --dataset DATASET_PATH --fps SOURCE_FPS
uv run main.py --dataset DATASET_PATH --fps SOURCE_FPS --objects-csv objects_seed.csv
uv run main.py --dataset DATASET_PATH --fps SOURCE_FPS --yaw-csv yaw.csv --objects-csv objects_seed.csv --object-frame world
```

`--no-video` is available for quick inspection, but the challenge submission
requires the MP4. A dataset folder contains `rgb/`, `xyz/`, and `bboxes_light.csv`.

## Validation still to perform

Tests and helper scripts are deferred until the Python implementation has been
reviewed, as requested. Next checks should cover coordinate signs, initial-axis
alignment, known yaw, invalid patches, unequal frame spacing, gap preservation,
track loss/reseeding, and a short real MP4 export. Then inspect first/middle/last
RGB frames, plot raw light measurements, and verify whether heading is stable.
Do not present an untested run or a synthetic sequence as challenge results.

## Actual-data run (format adaptation)

The supplied `data/` folder uses `rgb/leftNNNNNN.png` and
`xyz/depthNNNNNN.npz`. The loader now supports these IDs and either `points` or
`xyz` NPZ keys, with three coordinate channels and an optional fourth channel
that is discarded. The observed Y values are positive on the image left;
`--camera-y left` normalizes those values for the internal geometry.

For this dataset the reproducible command is:

```bash
uv run python main.py --dataset ./data --fps 30 --fps-note "Estimated from the README ten-second clip; unverified" --camera-y left --allow-missing-depth --light-box 365 447 400 496 --output ./outputs
```

The FPS is an estimate, not camera metadata. `--allow-missing-depth` preserves
all RGB frames but leaves absent XYZ observations missing. No long gaps are
interpolated. There is no supplied bounding-box CSV in this folder, so the light
is manually initialized and then tracked using red/green HSV components nearest
the last housing-center estimate. A bulb-size-based vertical offset accounts for
red being at the top and green at the bottom. This approximates the housing
center and introduces localization uncertainty. Derived boxes are saved as
`tracked_light_boxes.csv`; they are not supplied ground-truth annotations.

This produces a baseline trajectory, not a validated full-pose estimate. The
camera visibly changes heading, so constant-heading world positions can be
biased. Heading estimation remains necessary for an accurate turning trajectory.
