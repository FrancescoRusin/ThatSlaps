import functools
from dataclasses import dataclass
import mediapipe as mp
import time
import numpy
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from Utils import Action


CACHE_SIZE = 10


@dataclass
class HandSnapshot:
    wrist: tuple[float, float, float]
    middle: tuple[float, float, float]
    thumb: tuple[float, float, float]

    @functools.cached_property
    def center(self) -> tuple[float, float]:
        return (self.middle[0] + self.wrist[0]) / 2, (self.middle[1] + self.wrist[1]) / 2


class SlapTracker:
    def __init__(self):
        self.landmarker: mp.tasks.vision.hand_landmarker.HandLandmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path="models/hand_landmarker.task"),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            ))
        )
        self.archive: list[HandSnapshot] = []

    def reset(self):
        self.archive.clear()

    def detect_hand(self, image: numpy.ndarray) -> list[NormalizedLandmark] | None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        hand_landmarks: list[list[NormalizedLandmark]] = self.landmarker.detect_for_video(mp_image, int(time.time() * 1000)).hand_landmarks
        return hand_landmarks[0] if len(hand_landmarks) > 0 else None

    def check_slap(self, landmarks: list[NormalizedLandmark]) -> Action:
        if len(landmarks) > 0:
            self.archive.append(
                HandSnapshot(
                    wrist=(landmarks[0].x, landmarks[0].y, landmarks[0].z),
                    middle=(landmarks[12].x, landmarks[12].y, landmarks[12].z),
                    thumb=(landmarks[4].x, landmarks[4].y, landmarks[4].z)
                )
            )
        if len(self.archive) > CACHE_SIZE:
            self.archive.pop(0)
        else:
            return Action.NOTHING
        centers_x: list[float] = [p.center[0] for p in self.archive]
        centers_y: list[float] = [p.center[1] for p in self.archive]
        if numpy.std(centers_y) < 0.1:
            if all([x <= .45 for x in centers_x[:CACHE_SIZE // 2]]) and all([x >= .55 for x in centers_x[CACHE_SIZE // 2:]]):
                return Action.SLAP_RIGHT
            if all([x >= .55 for x in centers_x[:CACHE_SIZE // 2]]) and all([x <= .45 for x in centers_x[CACHE_SIZE // 2:]]):
                return Action.SLAP_LEFT
        elif numpy.std(centers_x) < 0.2:
            if all([y <= .5 for y in centers_y[:CACHE_SIZE // 2]]) and all([y >= .5 for y in centers_y[CACHE_SIZE // 2:]]):
                return Action.PET
        return Action.NOTHING

    def shutdown(self):
        self.landmarker.close()
