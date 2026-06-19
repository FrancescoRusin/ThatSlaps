"""Headless gesture-recognition API (slap left/right / pet / neutral).

Geometric hand-movement classification plus two ways to drive it without any UI:

    detect_gesture(...)  one-shot: open the webcam, capture a short bundle of
                         frames, and return a single label.
    GestureStream(...)   stateful: push frames one at a time and get a rolling
                         decision over the last `window` frames.

The gesture is decided geometrically from the hand bounding boxes (no VLM):

    slap_left  = horizontal movement toward the left of the frame
    slap_right = horizontal movement toward the right of the frame
    pet        = vertical movement of at least one hand
    neutral    = slight or no movement in either hand

Hand boxes come from the offline detectors in :mod:`src.Detectors`.
"""

import time
import statistics
from collections import deque, namedtuple

import numpy as np
import cv2

from src.Detectors import make_detector

# --- Gesture labels ---
# Horizontal movement is split by direction (in image coordinates, x grows to the
# right of the frame), matching Utils.Action: rightward -> SLAP_RIGHT.
SLAP_LEFT = "slap_left"      # horizontal movement toward the left of the frame
SLAP_RIGHT = "slap_right"    # horizontal movement toward the right of the frame
PET = "pet"                  # vertical movement
NEUTRAL = "neutral"          # little / no movement
STALL = "stall"            # (GestureStream) buffer not full yet; keep feeding frames

# Convenience: both slap directions, for callers that only care "was it a slap?".
SLAPS = frozenset({SLAP_LEFT, SLAP_RIGHT})

# What GestureStream.push returns each call: the rolling-window decision plus the
# detections found in the frame just pushed. `detections` is a list (possibly
# empty) of dicts {"bbox": (x1, y1, x2, y2), "center": (cx, cy), "label", "score"}
# in pixel coordinates -- so result.detections[i]["bbox"] is a hand's box.
GestureResult = namedtuple("GestureResult", ["decision", "detections"])

# --- Capture / bundling ---
DETECTOR_BACKEND = "mediapipe"        # default detector backend
CAPTURE_FPS = 30                      # frames sampled into a bundle per second
FRAMES_PER_BATCH = 30                 # bundle / window size (~1s of motion)
CAPTURE_WIDTH, CAPTURE_HEIGHT = 1280, 720

# --- Movement decision tuning ---
# Movement is measured in "hand-widths" (the per-track sweep divided by the
# hand's own size), so a hand near or far from the camera behaves the same. A
# gesture is a clear sweep of the hand across the view; small jitter never covers
# that distance and stays neutral. Fast, brief moves are caught because they
# still sweep far -- short tracks are kept (not filtered on presence) so a
# movement spanning only a few frames still counts.
MOVE_THRESHOLD = 0.8      # min sweep (hand-widths) to count as a gesture
MIN_TRACK_FRAMES = 3      # a track needs >= this many detections to be scored
MAX_JUMP_FACTOR = 2.5     # nearest-center association gate (x hand size)


# ----------------------------------------------------------------------------
# Movement classification
# ----------------------------------------------------------------------------
def _hand_size(det):
    """Characteristic size of a detection (max of bbox width/height), in pixels."""
    x1, y1, x2, y2 = det["bbox"]
    return max(x2 - x1, y2 - y1)


def _build_tracks(per_frame_detections):
    """Group detections across frames into per-hand tracks.

    Uses handedness labels when the detector provides them (MediaPipe); otherwise
    falls back to greedy nearest-center association (YOLO). Returns a list of
    tracks, each a list of detection dicts in frame order.
    """
    # Find the first non-empty frame to decide whether labels are available.
    first = next((dets for dets in per_frame_detections if dets), None)
    if first is None:
        return []
    use_labels = first[0]["label"] is not None

    if use_labels:
        grouped = {}
        for dets in per_frame_detections:
            for d in dets:
                grouped.setdefault(d["label"], []).append(d)
        return list(grouped.values())

    # Nearest-center association across consecutive frames.
    tracks = []          # list of {"dets": [...], "center": (cx, cy)}
    for dets in per_frame_detections:
        used = set()
        # Extend existing tracks with their nearest unused detection.
        for tr in tracks:
            best_j, best_dist = None, None
            for j, d in enumerate(dets):
                if j in used:
                    continue
                dist = np.hypot(d["center"][0] - tr["center"][0],
                                d["center"][1] - tr["center"][1])
                if best_dist is None or dist < best_dist:
                    best_j, best_dist = j, dist
            if best_j is not None:
                gate = MAX_JUMP_FACTOR * _hand_size(dets[best_j])
                if best_dist <= gate:
                    d = dets[best_j]
                    tr["dets"].append(d)
                    tr["center"] = d["center"]
                    used.add(best_j)
        # Unmatched detections start new tracks.
        for j, d in enumerate(dets):
            if j not in used:
                tracks.append({"dets": [d], "center": d["center"]})

    # Keep at most the two longest tracks.
    tracks.sort(key=lambda t: len(t["dets"]), reverse=True)
    return [t["dets"] for t in tracks[:2]]


def classify_from_detections(per_frame_detections):
    """Decide slap_left / slap_right / pet / neutral from per-frame boxes.

    A gesture is a clear sweep of a hand across the view along its dominant axis
    (>= MOVE_THRESHOLD hand-widths). Short tracks are kept, so a quick movement
    spanning only a few frames still counts; small jitter -- which never covers
    that distance -- stays neutral.

    Horizontal sweeps are split by direction: the side (min or max x) the hand
    reaches *later* in time is where it ended up, so reaching max x last means it
    moved right (SLAP_RIGHT) and reaching min x last means it moved left
    (SLAP_LEFT). Vertical sweeps return PET.
    """
    if not per_frame_detections:
        return NEUTRAL

    best_sweep, best_label = 0.0, None

    for track in _build_tracks(per_frame_detections):
        if len(track) < MIN_TRACK_FRAMES:
            continue
        xs = [d["center"][0] for d in track]
        ys = [d["center"][1] for d in track]

        scale = statistics.median(_hand_size(d) for d in track)
        if scale <= 0:
            continue

        # Dominant axis via summed per-step motion (robust to back-and-forth).
        dx_total = sum(abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1))
        dy_total = sum(abs(ys[i + 1] - ys[i]) for i in range(len(ys) - 1))
        horizontal = dx_total >= dy_total
        vals = xs if horizontal else ys

        sweep = (max(vals) - min(vals)) / scale
        if sweep <= best_sweep:
            continue
        best_sweep = sweep
        if horizontal:
            rightward = vals.index(max(vals)) > vals.index(min(vals))
            best_label = SLAP_RIGHT if rightward else SLAP_LEFT
        else:
            best_label = PET

    if best_label is None or best_sweep < MOVE_THRESHOLD:
        return NEUTRAL
    return best_label


def classify_movement(frames, detector):
    """Classify a bundle of frames as slap_left / slap_right / pet / neutral.

    Runs the detector on each frame, tracks the hand boxes, and returns one of
    the labels.
    """
    per_frame_detections = [detector.detect(f) for f in frames]
    return classify_from_detections(per_frame_detections)


# ----------------------------------------------------------------------------
# One-shot capture API
# ----------------------------------------------------------------------------
def detect_gesture(detector=None, backend=DETECTOR_BACKEND, weights=None,
                   mp_model=None, camera=0, num_frames=FRAMES_PER_BATCH,
                   fps=CAPTURE_FPS, warmup_frames=3):
    """Capture one bundle of webcam frames and return the gesture label.

    Fully headless: opens the camera, samples ``num_frames`` frames at ``fps``,
    runs the hand detector on each, classifies the movement, and returns one of
    ``SLAP_LEFT`` / ``SLAP_RIGHT`` / ``PET`` / ``NEUTRAL``. No window is shown and
    no keys are read.

    Pass a pre-built ``detector`` to reuse it across calls (avoids reloading the
    model each time); otherwise one is created from ``backend`` / ``weights`` /
    ``mp_model`` and closed again before returning. ``warmup_frames`` are grabbed
    and discarded first so auto-exposure can settle and the camera buffer is
    fresh.

    Raises RuntimeError if the camera can't be opened. Returns ``NEUTRAL`` if no
    frames could be grabbed.
    """
    own_detector = detector is None
    if own_detector:
        detector = make_detector(backend, weights, mp_model)

    cap = cv2.VideoCapture(camera)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

        for _ in range(warmup_frames):
            cap.read()

        sample_interval = 1.0 / fps
        frames = []
        last_sample_time = 0.0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            if now - last_sample_time >= sample_interval:
                last_sample_time = now
                frames.append(frame)

        return classify_movement(frames, detector)
    finally:
        cap.release()
        if own_detector:
            detector.close()


# ----------------------------------------------------------------------------
# Streaming API
# ----------------------------------------------------------------------------
class GestureStream:
    """Stateful, headless gesture classifier fed one frame at a time.

    Unlike :func:`detect_gesture`, this does no video capture: the caller pushes
    individual OpenCV BGR images (e.g. from their own camera/render loop) via
    :meth:`push`. Each pushed frame is run through the detector once and its
    detections are cached in a rolling buffer of the last ``window`` frames.

    Every :meth:`push` returns a :class:`GestureResult` ``(decision, detections)``
    where ``detections`` are the hand boxes found in that very frame and
    ``decision`` is based on the buffered window:

        * ``STALL``   - fewer than ``window`` frames buffered so far; keep going.
        * otherwise   - one of ``SLAP_LEFT`` / ``SLAP_RIGHT`` / ``PET`` /
                         ``NEUTRAL`` for the last ``window`` frames.

    Pass a pre-built ``detector`` to reuse it, or let one be created from
    ``backend`` / ``weights`` / ``mp_model`` (closed by :meth:`close` only if
    this object created it). Usable as a context manager.
    """

    def __init__(self, detector=None, backend=DETECTOR_BACKEND, weights=None,
                 mp_model=None, window=FRAMES_PER_BATCH):
        if window < 2:
            raise ValueError("window must be at least 2 frames")
        self._own_detector = detector is None
        self.detector = detector or make_detector(backend, weights, mp_model)
        self.window = window
        # Cache detections (not raw frames): detect once per pushed frame, so a
        # full-window classification is just a lookup, never a re-detection.
        self._buffer = deque(maxlen=window)

    def push(self, image):
        """Add one BGR frame and return a :class:`GestureResult`.

        ``result.detections`` is this frame's hand boxes (a list, possibly empty;
        each entry has a ``"bbox"`` of ``(x1, y1, x2, y2)`` in pixels). It is
        returned on every call, including while stalling.

        ``result.decision`` is ``STALL`` until ``window`` frames have been
        buffered, then one of ``SLAP_LEFT`` / ``SLAP_RIGHT`` / ``PET`` /
        ``NEUTRAL`` on every subsequent call (the buffer slides, always
        reflecting the most recent ``window`` frames).
        """
        detections = self.detector.detect(image)
        self._buffer.append(detections)
        if len(self._buffer) < self.window:
            decision = STALL
        else:
            decision = classify_from_detections(list(self._buffer))
        return GestureResult(decision, detections)

    def reset(self):
        """Clear the buffer (next decisions STALL until it refills)."""
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

    def close(self):
        """Release the detector if this object created it."""
        if self._own_detector:
            self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
