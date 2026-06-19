"""Offline hand detectors used by the geometric gesture classifier.

Each detector exposes a single method:

    detector.detect(frame_bgr) -> list of detections

where a detection is a dict in pixel coordinates::

    {
        "bbox":   (x1, y1, x2, y2),
        "center": (cx, cy),
        "label":  "Left" | "Right" | None,   # handedness, if the backend has it
        "score":  float,
    }

Two interchangeable backends are provided and selected via :func:`make_detector`:

    mediapipe : Google MediaPipe Tasks HandLandmarker. The small model file is
                auto-downloaded once and cached locally, then runs fully offline.
                Gives 21 landmarks + Left/Right handedness.
    yolo      : Ultralytics YOLO with a hand-trained .pt (supply the weights path).

Heavy backend libraries (mediapipe, ultralytics) are imported lazily inside each
detector, so importing this module is cheap and only the chosen backend needs to
be installed.
"""

import os
import urllib.request

import cv2


class MediaPipeHandDetector:
    """Hand boxes from MediaPipe's Tasks HandLandmarker (offline after one-time
    model download). Recent mediapipe wheels (0.10.x) ship only the Tasks API,
    which needs a small `hand_landmarker.task` file. It is fetched once and
    cached next to this module, then everything runs offline.
    """

    MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                 "hand_landmarker/float16/1/hand_landmarker.task")
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "hand_landmarker.task")

    def __init__(self, model_path=None, max_hands=2, min_conf=0.5):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise ImportError(
                "mediapipe is not installed. Run: pip install mediapipe"
            ) from e
        self._mp = mp

        model_path = model_path or self.DEFAULT_MODEL_PATH
        if not os.path.exists(model_path):
            print(f"[MediaPipe] Downloading hand model (one-time) -> {model_path}")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        detections = []
        hands_lms = result.hand_landmarks or []
        handedness = result.handedness or [None] * len(hands_lms)
        for lms, hand in zip(hands_lms, handedness):
            xs = [lm.x * w for lm in lms]
            ys = [lm.y * h for lm in lms]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            label, score = None, 1.0
            if hand:
                label = hand[0].category_name          # "Left" / "Right"
                score = float(hand[0].score)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "label": label,
                "score": score,
            })
        return detections

    def close(self):
        self._landmarker.close()


class YoloHandDetector:
    """Hand boxes from an Ultralytics YOLO hand model (requires a local .pt)."""

    def __init__(self, weights, conf=0.25):
        if not weights:
            raise ValueError(
                "The 'yolo' backend needs a hand-trained weights file. "
                "Pass it with the weights path (generic COCO YOLO has no "
                "'hand' class)."
            )
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from e
        self._model = YOLO(weights)
        self._conf = conf

    def detect(self, frame_bgr):
        results = self._model.predict(frame_bgr, conf=self._conf, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "label": None,  # YOLO gives no handedness
                    "score": float(box.conf[0].item()),
                })
        return detections

    def close(self):
        pass


def make_detector(backend, weights=None, mp_model=None):
    """Factory: return a detector for the chosen backend.

    backend  : "mediapipe" or "yolo".
    weights  : path to a hand-trained YOLO .pt (required for the yolo backend).
    mp_model : path to a hand_landmarker.task (mediapipe); auto-downloaded if
               omitted.
    """
    backend = backend.lower()
    if backend == "mediapipe":
        return MediaPipeHandDetector(model_path=mp_model)
    if backend == "yolo":
        return YoloHandDetector(weights)
    raise ValueError(f"Unknown detector backend: {backend!r} (use 'mediapipe' or 'yolo')")
