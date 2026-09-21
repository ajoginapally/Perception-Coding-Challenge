"""Part B: seed/reseed RGB boxes and follow object features across frames.

A seed CSV has frame_id,object_id,label,x_min,y_min,x_max,y_max. Additional
rows for an existing ID reinitialize it after occlusion or drift. No detector
is assumed to recognize a golf cart correctly without inspecting the dataset.
"""
from dataclasses import dataclass
import csv
import cv2
import numpy as np
from .data import frame_id


@dataclass
class Track:
    object_id: str
    label: str
    box: np.ndarray
    features: np.ndarray | None
    status: str = "seeded"
    inliers: int = 0


def load_seeds(path):
    if path is None:
        return {}
    result = {}
    seen = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = {"frame_id", "object_id", "label", "x_min", "y_min", "x_max", "y_max"}
        if not columns <= set(reader.fieldnames or []):
            raise ValueError(f"Object CSV needs {sorted(columns)}")
        for row in reader:
            fid = frame_id(row["frame_id"])
            identity = row["object_id"].strip()
            label = row["label"].strip()
            box = np.array([float(row[k]) for k in ("x_min", "y_min", "x_max", "y_max")])
            if not identity or not label or (fid, identity) in seen:
                raise ValueError(f"Empty label/ID or duplicate seed in frame {fid}")
            if not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                raise ValueError(f"Invalid object seed: {row}")
            seen.add((fid, identity))
            result.setdefault(fid, []).append((identity, label, box))
    return result


def features_in_box(gray, box):
    h, w = gray.shape
    x1, y1 = np.maximum(np.floor(box[:2]).astype(int), 0)
    x2, y2 = np.minimum(np.ceil(box[2:]).astype(int), [w, h])
    mask = np.zeros_like(gray)
    mask[y1:y2, x1:x2] = 255
    return cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=100,
                                   qualityLevel=0.02, minDistance=5, blockSize=5)


class ObjectTracker:
    """LK forward/backward check + RANSAC similarity fit, with explicit loss.

    Lost tracks are not silently held at their last location or extrapolated.
    They need another seed row. Boxes cannot identify foreground perfectly;
    overlays and reseeding remain important for background-heavy regions.
    """
    def __init__(self):
        self.previous_gray = None
        self.tracks = {}

    def update(self, image, seeds):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if self.previous_gray is not None and self.previous_gray.shape != gray.shape:
            raise ValueError("Image dimensions changed during tracking")
        for track in self.tracks.values():
            if track.status == "lost":
                continue
            old = track.features
            track.status = "lost"
            track.inliers = 0
            if old is None or len(old) < 6:
                continue
            new, forward, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, old, None)
            if new is None:
                continue
            back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, new, None)
            if back is None:
                continue
            keep = (forward.ravel() == 1) & (backward.ravel() == 1)
            keep &= np.linalg.norm((back-old).reshape(-1,2), axis=1) < 1.5
            keep &= np.isfinite(new.reshape(-1,2)).all(axis=1)
            a, b = old.reshape(-1,2)[keep], new.reshape(-1,2)[keep]
            if len(a) < 6:
                continue
            transform, inliers = cv2.estimateAffinePartial2D(
                a, b, method=cv2.RANSAC, ransacReprojThreshold=2.5)
            if transform is None or inliers is None or inliers.sum() < 6:
                continue
            if inliers.mean() < 0.5:
                continue
            scale = np.linalg.norm(transform[:,0])
            if not 0.75 <= scale <= 1.33:
                continue
            x1, y1, x2, y2 = track.box
            corners = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]])
            warped = corners @ transform[:,:2].T + transform[:,2]
            box = np.r_[warped.min(axis=0), warped.max(axis=0)]
            box[:2] = np.maximum(box[:2], [0,0])
            box[2:] = np.minimum(box[2:], [w,h])
            if np.any(box[2:] - box[:2] < 4):
                continue
            track.box, track.status = box, "tracked"
            track.inliers = int(inliers.sum())
            track.features = features_in_box(gray, box)
        for identity, label, box in seeds:
            if np.any(box[:2] < 0) or np.any(box[2:] > [w,h]):
                raise ValueError(f"Seed {identity} extends outside the image")
            self.tracks[identity] = Track(identity, label, box.copy(), features_in_box(gray, box))
        self.previous_gray = gray
        return list(self.tracks.values())


class SignalColorTracker:
    """Follow a selected illuminated signal using red/green connected components.

    Select the component nearest the previous housing center. Compensate for
    red being the top bulb and green the bottom, using measured bulb diameter.
    This is a visible-housing-center approximation, not a fixed surveyed point.
    """
    def __init__(self):
        self.center = None
        self.diameter = None

    def update(self, image, initial_box=None):
        if initial_box is not None:
            self.center = (np.asarray(initial_box[:2])+initial_box[2:])/2
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        candidates = []
        for name, mask in [("red", ((h < 12)|(h > 170))), ("green", ((h > 65)&(h < 105)))]:
            mask = (mask & (s > 120) & (v > 180)).astype(np.uint8)*255
            contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area=cv2.contourArea(contour)
                x,y,w,hh=cv2.boundingRect(contour)
                if area < 8 or area > 2500 or not 0.5 < w/hh < 2 or y > image.shape[0]*0.65:
                    continue
                diameter=(w+hh)/2
                center=np.array([x+w/2, y+hh/2+(1.1*diameter if name=="red" else -1.1*diameter)])
                distance=np.linalg.norm(center-self.center)
                if distance < 85:
                    candidates.append((distance,center,diameter))
        if not candidates:
            return None
        _,self.center,diameter = min(candidates,key=lambda c:c[0])
        self.diameter = diameter if self.diameter is None else 0.8*self.diameter+0.2*diameter
        half = np.array([1.1*self.diameter,1.8*self.diameter])
        return np.r_[self.center-half,self.center+half]
