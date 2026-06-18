import math
import time
from enum import Enum, auto
import sdl2
import sdl2.ext
import sdl2.sdlttf

SCREEN_W: int = 1920
SCREEN_H: int = 1080
FRAMERATE: int = 30
MAX_TRIALS: int = 20
LOAD_TIME: int = 30
CHOICE_SECONDS: int = 3


class Mode(Enum):
    LOAD = auto()
    CHOICE = auto()
    CONSEQUENCE = auto()


class Target(Enum):
    UGLY = auto()
    CUTE = auto()
    DANGER = auto()


class Action(Enum):
    SLAP_LEFT = auto()
    SLAP_RIGHT = auto()
    PET = auto()
    NOTHING = auto()


def detect_action():
    pass


def should_close() -> bool:
    events = sdl2.ext.get_events()
    for event in events:
        if (event.type == sdl2.SDL_QUIT or
                (event.type == sdl2.SDL_KEYUP and (event.key.keysym.sym == sdl2.SDLK_ESCAPE or event.key.keysym.sym == sdl2.SDLK_w))):
            return True
    return False


def time_int() -> int:
    return time.time_ns() // int(1e6)


def update_time_texture(renderer, texture, font, time_displayed, color):
    sdl2.SDL_DestroyTexture(texture)
    text_surface = sdl2.sdlttf.TTF_RenderUTF8_Blended(font, f"{time_displayed}".encode("utf-8"), color)
    new_texture = sdl2.SDL_CreateTextureFromSurface(renderer.sdlrenderer, text_surface)
    sdl2.SDL_FreeSurface(text_surface)
    return new_texture
