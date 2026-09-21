"""Static and video output with fixed meter axes and visible missing data."""
import os
from pathlib import Path
import cv2
import numpy as np


def setup_plotting(output):
    # Headless output also works over SSH; avoid a global user cache write.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(output).resolve() / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def bounds(points):
    points = np.asarray(points).reshape(-1,2)
    points = points[np.isfinite(points).all(axis=1)]
    points = np.vstack([points, [0,0]])
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = max(2.0, float(np.max(hi-lo))*0.1)
    return lo-pad, hi+pad


def configure_axis(ax, limits, ego=False):
    lo, hi = limits
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
           xlabel="Forward X (m)" if ego else "World X (m)",
           ylabel="Left Y (m)" if ego else "World Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)


def render_outputs(output, frames, times, raw, positions, angles, objects, fps,
                   light_boxes, make_video=True, object_frame="ego", mode="constant heading"):
    output = Path(output)
    plt = setup_plotting(output)
    limits = bounds(np.vstack([raw, positions]))
    valid = np.flatnonzero(np.isfinite(positions).all(axis=1))
    fig, ax = plt.subplots(figsize=(8,6), layout="constrained")
    configure_axis(ax, limits)
    
    
    ax.plot(raw[:,0], raw[:,1], ".", alpha=0.35, color="gray", label="Raw measurement")
    ax.plot(positions[:,0], positions[:,1], ".-", lw=1, label="Ego trajectory")
    ax.scatter(0,0, marker="*", s=150, color="orange", label="Traffic light")
    ax.scatter(*positions[valid[0]], marker="s", label="Start")
    ax.scatter(*positions[valid[-1]], marker="X", label="End")
    ax.set_title(f"Traffic-light anchored trajectory — {mode}")
    ax.legend()
    fig.savefig(output / "trajectory.png", dpi=180)
    plt.close(fig)
    
    
    if not make_video:
        return
    import imageio_ffmpeg
    import matplotlib
    from matplotlib.animation import FFMpegWriter
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    # Part A always uses the requested fixed world frame.
    fig, ax = plt.subplots(figsize=(8,6), layout="constrained")
    configure_axis(ax, limits)
    ax.scatter(0,0, marker="*", s=150, color="orange", label="Traffic light")
    line, = ax.plot([], [], ".-", lw=1, label="Ego path")
    current = ax.scatter([], [], s=70, label="Current ego")
    title = ax.set_title("")
    ax.legend()
    with FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]).saving(
            fig, str(output / "trajectory.mp4"), dpi=100) as writer:
        for k in range(len(frames)):
            line.set_data(positions[:k+1,0], positions[:k+1,1])
            current.set_offsets(positions[k:k+1] if np.isfinite(positions[k]).all() else np.empty((0,2)))
            status = "" if np.isfinite(positions[k]).all() else " | missing position"
            title.set_text(f"t={times[k]:.2f}s | {mode}{status}")
            repeats = frames[k+1][0]-frames[k][0] if k+1 < len(frames) else 1
            for _ in range(repeats):
                writer.grab_frame()
    plt.close(fig)
    if not objects:
        return

    # Rich Part B video includes RGB evidence of the tracking and sample boxes.
    by_frame = {}
    identities = sorted({row["object_id"] for row in objects})
    colors = {identity: plt.get_cmap("tab10")(i % 10) for i, identity in enumerate(identities)}
    coordinate_keys = ("ego_x", "ego_y") if object_frame == "ego" else ("world_x", "world_y")
    mapped = [[row[coordinate_keys[0]], row[coordinate_keys[1]]] for row in objects]
    scene_limits = bounds(mapped + (positions.tolist() if object_frame == "world" else [[0,0]]))
    for row in objects:
        by_frame.setdefault(row["frame_id"], []).append(row)
    fig, (rgb_ax, bev_ax) = plt.subplots(1,2, figsize=(14,6), layout="constrained")
    with FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]).saving(
            fig, str(output / "objects_bev.mp4"), dpi=100) as writer:
        for k, (fid, image_path, _, _) in enumerate(frames):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read {image_path}")
            light = light_boxes[k]
            if light is not None:
                a,b,c,d = np.rint(light).astype(int)
                cv2.rectangle(image, (a,b), (c,d), (0,255,255), 2)
                cv2.circle(image, ((a+c)//2,(b+d)//2), 3, (0,255,255), -1)
            rgb_ax.clear()
            bev_ax.clear()
            configure_axis(bev_ax, scene_limits, object_frame == "ego")
            if object_frame == "world":
                bev_ax.plot(positions[:k+1,0], positions[:k+1,1], color="steelblue", alpha=0.5)
                if np.isfinite(positions[k]).all():
                    bev_ax.scatter(*positions[k], marker="^", color="black", label="Ego")
                bev_ax.scatter(0,0,marker="*",color="orange",label="Light")
            else:
                bev_ax.scatter(0,0,marker="^",color="black",label="Ego")
            for row in by_frame.get(fid, []):
                if row["tracking_status"] == "lost":
                    continue
                identity = row["object_id"]
                a,b,c,d = [int(round(row[key])) for key in ("x_min","y_min","x_max","y_max")]
                color = colors[identity]
                bgr = tuple(int(255*x) for x in color[:3][::-1])
                cv2.rectangle(image, (a,b), (c,d), bgr, 2)
                text = f"{identity}: {row['label']} ({row['depth_status']})"
                cv2.putText(image,text,(a,max(15,b-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,bgr,1)
                xy = np.array([row[key] for key in coordinate_keys])
                if np.isfinite(xy).all():
                    bev_ax.scatter(*xy, color=color, label=f"{identity}: {row['label']}")
            rgb_ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            rgb_ax.axis("off")
            rgb_ax.set_title(f"Frame {fid} | t={times[k]:.2f}s")
            bev_ax.set_title(f"{object_frame.capitalize()}-frame objects")
            bev_ax.legend(fontsize=8, loc="upper right")
            repeats = frames[k+1][0]-fid if k+1 < len(frames) else 1
            for _ in range(repeats):
                writer.grab_frame()
    plt.close(fig)
