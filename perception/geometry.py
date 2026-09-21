"""Camera forward/right/up -> level world forward/left/up geometry."""
import numpy as np


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def local_xy(xyz):
    return np.asarray(xyz)[..., :2] * np.array([1.0, -1.0])


def reference_angle(initial_xyz):
    q = local_xy(initial_xyz)
    return -np.arctan2(q[1], q[0])


def camera_position(light_xyz, angle):
    return -(rotation(angle) @ local_xy(light_xyz))


def object_position(object_xyz, camera_xy, angle):
    return camera_xy + rotation(angle) @ local_xy(object_xyz)


def fill_short_gaps(values, max_gap, times=None):
    result = np.asarray(values, dtype=float).copy()
    times = np.arange(len(result), dtype=float) if times is None else np.asarray(times)
    interpolated = np.zeros(len(result), dtype=bool)
    good = np.isfinite(result).all(axis=1)
    indices = np.flatnonzero(good)
    for left, right in zip(indices[:-1], indices[1:]):
        gap = right - left - 1
        if 0 < gap <= max_gap:
            for k in range(left + 1, right):
                f = (times[k] - times[left]) / (times[right] - times[left])
                result[k] = (1 - f) * result[left] + f * result[right]
                interpolated[k] = True
    return result, interpolated


def smooth_segments(values, window):
    result = np.asarray(values, dtype=float).copy()
    if window <= 1:
        return result
    good = np.isfinite(result).all(axis=1)
    boundaries = np.flatnonzero(np.diff(np.r_[False, good, False]))
    half = window // 2
    for start, end in zip(boundaries[::2], boundaries[1::2]):
        for k in range(start, end):
            result[k] = np.mean(values[max(start, k-half):min(end, k+half+1)], axis=0)
    return result
