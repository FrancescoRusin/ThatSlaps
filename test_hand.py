"""Offline hand-movement gesture classifier (slap / pet / neutral).

A lightweight, fully-local alternative to the VLM pipeline in test.py. Instead of
asking a vision-language model what it sees, this captures a short bundle of
frames, tracks the hand bounding boxes across them, and decides the gesture
geometrically:

    slap    = horizontal movement of at least one hand
    pet     = vertical   movement of at least one hand
    neutral = slight or no movement in either hand

Hand boxes come from a local AI model that runs offline on macOS and Windows.
Two interchangeable backends are supported, chosen with --detector:

    mediapipe : Google MediaPipe Tasks HandLandmarker. The small model file is
                auto-downloaded once and cached locally, then runs fully offline.
                Gives 21 landmarks + Left/Right handedness.
    yolo      : Ultralytics YOLO with a hand-trained .pt (supply via --weights).

Usage:
    pip install mediapipe                 # for the default backend
    python test_yolo.py --detector mediapipe
    python test_yolo.py --detector yolo --weights hand_yolov8.pt
"""

import time
import sys
import argparse
import statistics
import numpy as np
import cv2

from src.Detectors import make_detector

# --- Gesture labels (returned by classify_movement) ---
SLAP = "slap"        # horizontal movement
PET = "pet"          # vertical movement
NEUTRAL = "neutral"  # little / no movement

# --- Capture / bundling (mirrors test.py) ---
DETECTOR_BACKEND = "mediapipe"        # default; overridable with --detector
CAPTURE_FPS = 10                      # frames sampled into a bundle per second
FRAMES_PER_BATCH = 10                 # bundle size (~1s of motion at 10 fps)
SAMPLE_INTERVAL = 1.0 / CAPTURE_FPS   # seconds between sampled frames
CAPTURE_WIDTH, CAPTURE_HEIGHT = 1280, 720
DURATION_SECONDS = 60

# --- Movement decision tuning ---
# Movement is measured in "hand-widths" (the per-track range divided by the
# hand's own size), so a hand near or far from the camera behaves the same.
MOVE_THRESHOLD = 0.6      # min amplitude (hand-widths) to count as a gesture
MIN_DET_FRACTION = 0.5    # a track must be seen in >= this fraction of frames
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
    """Decide slap / pet / neutral from already-detected per-frame boxes."""
    num_frames = len(per_frame_detections)
    if num_frames == 0:
        return NEUTRAL

    min_count = max(2, int(np.ceil(MIN_DET_FRACTION * num_frames)))
    best_amp, best_axis = 0.0, None

    for track in _build_tracks(per_frame_detections):
        if len(track) < min_count:
            continue
        centers = [d["center"] for d in track]
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]

        # Dominant axis via summed per-step motion (robust to back-and-forth).
        dx_total = sum(abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1))
        dy_total = sum(abs(ys[i + 1] - ys[i]) for i in range(len(ys) - 1))

        scale = statistics.median(_hand_size(d) for d in track)
        if scale <= 0:
            continue

        horizontal = dx_total >= dy_total
        amplitude = ((max(xs) - min(xs)) if horizontal else (max(ys) - min(ys))) / scale

        if amplitude > best_amp:
            best_amp, best_axis = amplitude, ("h" if horizontal else "v")

    if best_axis is None or best_amp < MOVE_THRESHOLD:
        return NEUTRAL
    return SLAP if best_axis == "h" else PET


def classify_movement(frames, detector):
    """Classify a bundle of frames as slap / pet / neutral.

    Runs the detector on each frame, tracks the hand boxes, and returns one of
    the three labels. This is the standalone entry point; the live demo loop in
    main() uses classify_from_detections to avoid detecting each frame twice.
    """
    per_frame_detections = [detector.detect(f) for f in frames]
    return classify_from_detections(per_frame_detections)


# ----------------------------------------------------------------------------
# Live preview (style borrowed from test.py)
# ----------------------------------------------------------------------------
def draw_overlay(frame, detections, decision, time_remaining, buffer_count):
    """Return a copy of frame with hand boxes and the current decision drawn."""
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if d["label"]:
            cv2.putText(out, d["label"], (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    cv2.putText(out, f"Decision: {decision.upper()}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(out, f"Time left: {int(time_remaining)}s | Buffer: {buffer_count}/{FRAMES_PER_BATCH}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description="Offline hand-movement gesture classifier.")
    parser.add_argument("--detector", choices=["mediapipe", "yolo"], default=DETECTOR_BACKEND,
                        help="Local hand-detection backend (default: mediapipe).")
    parser.add_argument("--weights", default=None,
                        help="Path to a hand-trained YOLO .pt (required for --detector yolo).")
    parser.add_argument("--mp-model", default=None,
                        help="Path to a hand_landmarker.task (mediapipe); auto-downloaded if omitted.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--duration", type=int, default=DURATION_SECONDS,
                        help="Run time in seconds (default: %(default)s).")
    args = parser.parse_args()

    try:
        detector = make_detector(args.detector, args.weights, args.mp_model)
    except (ImportError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"[Main] Using '{args.detector}' detector.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    print(f"[Main] Capture {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}. "
          f"Bundling {FRAMES_PER_BATCH} frames @ {CAPTURE_FPS}fps. Press 'q' to quit.")

    decision = NEUTRAL
    det_buffer = []          # detections for each sampled frame in the current bundle
    last_detections = []     # most recent frame's detections (for live boxes)
    last_sample_time = 0.0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        now = time.time()
        # Sample at a fixed rate: detect once per sampled frame (for both the
        # live boxes and the bundle), not on every camera frame.
        if now - last_sample_time >= SAMPLE_INTERVAL:
            last_sample_time = now
            last_detections = detector.detect(frame)
            det_buffer.append(last_detections)

            if len(det_buffer) == FRAMES_PER_BATCH:
                decision = classify_from_detections(det_buffer)
                print(f"[Main] Decision: {decision}")
                det_buffer = []

        time_remaining = max(0, args.duration - (now - start_time))
        cv2.imshow("Hand Gesture (offline)",
                   draw_overlay(frame, last_detections, decision, time_remaining, len(det_buffer)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User exited manually.")
            break
        if now - start_time > args.duration:
            print("Reached time limit. Exiting.")
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
