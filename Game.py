import time
import random

import sdl2.ext
import sdl2.sdlttf
import sdl2.sdlmixer
import cv2
import sys

from Utils import FRAMERATE, MAX_TRIALS, CHOICE_SECONDS, Mode, Target, Action, should_close, time_int
from Hands_mp import SlapTracker

if __name__ == "__main__":
    sdl2.ext.init(sdl2.SDL_INIT_VIDEO)
    sdl2.sdlttf.TTF_Init()
    mode = sdl2.SDL_DisplayMode()
    sdl2.SDL_GetCurrentDisplayMode(0, mode)
    SCREEN_W = mode.w
    SCREEN_H = mode.h
    window = sdl2.ext.Window("GAME", size=(SCREEN_W, SCREEN_H), flags=sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP)
    renderer = sdl2.ext.Renderer(window)

    red = sdl2.ext.Color(255, 0, 0)
    green = sdl2.ext.Color(0, 255, 0)
    blue = sdl2.ext.Color(0, 0, 255)
    black = sdl2.ext.Color(0, 0, 0)
    white = sdl2.SDL_Color(255, 255, 255)

    factory = sdl2.ext.SpriteFactory(renderer=renderer)
    sprite_factory = factory.create_sprite_render_system(window)

    font = sdl2.sdlttf.TTF_OpenFont(b"./fonts/Super Pandora.ttf", 500)

    webcam = cv2.VideoCapture(0)
    success, image = webcam.read()
    if not success:
        sys.exit(-1)
    height, width, channels = image.shape
    webcam_texture = sdl2.SDL_CreateTexture(renderer.sdlrenderer, sdl2.SDL_PIXELFORMAT_RGB24, sdl2.SDL_TEXTUREACCESS_STATIC, width, height)
    slap_checker: SlapTracker = SlapTracker()
    background_rect = sdl2.SDL_Rect(0, 0, SCREEN_W, SCREEN_H)
    text_rect = sdl2.SDL_Rect(SCREEN_W // 2 - 250, SCREEN_H // 2 - 250, 500, 500)
    timer_rect = sdl2.SDL_Rect(SCREEN_W // 2 - 100, 100, 200, 200)
    webcam_rect = sdl2.SDL_Rect(SCREEN_W - 300, SCREEN_H - 200, 300, 200)
    image_rect = sdl2.SDL_Rect(SCREEN_W // 2 - 250, SCREEN_H - 500, 500, 500)
    running: bool = True
    time_displayed: int = CHOICE_SECONDS
    renderer.color = black
    prev_tick: int = time_int()
    for n in range(3, 0, -1):
        text_surface = sdl2.sdlttf.TTF_RenderUTF8_Blended(font, f"{n}".encode("utf-8"), white)
        texture = sdl2.SDL_CreateTextureFromSurface(renderer.sdlrenderer, text_surface)
        sdl2.SDL_FreeSurface(text_surface)
        for _ in range(FRAMERATE):
            renderer.clear()
            sdl2.SDL_RenderCopy(renderer.sdlrenderer, texture, None, text_rect)
            current_tick: int = time_int()
            if (current_tick - prev_tick) * FRAMERATE < 1000:
                time.sleep((1000 / FRAMERATE - current_tick + prev_tick) / 1000)
            renderer.present()
            prev_tick = time_int()
            if should_close():
                running = False
                break
        sdl2.SDL_DestroyTexture(texture)
        if not running:
            break
    trials: int = 0
    game_mode: Mode = Mode.LOAD
    chosen_action: Action = Action.NOTHING
    timer_textures = []
    for n in range(1, CHOICE_SECONDS + 1):
        text_surface = sdl2.sdlttf.TTF_RenderUTF8_Blended(font, f"{n}".encode("utf-8"), white)
        timer_textures.append(sdl2.SDL_CreateTextureFromSurface(renderer.sdlrenderer, text_surface))
        sdl2.SDL_FreeSurface(text_surface)
    actions_textures = {}
    for a in Action:
        text_surface = sdl2.sdlttf.TTF_RenderUTF8_Blended(font, f"{a}".encode("utf-8"), white)
        actions_textures[a.value] = sdl2.SDL_CreateTextureFromSurface(renderer.sdlrenderer, text_surface)
        sdl2.SDL_FreeSurface(text_surface)
    sprite_background = factory.from_image("images/Arena-01.png")
    sdl2.SDL_SetTextureBlendMode(sprite_background.texture, sdl2.SDL_BLENDMODE_BLEND)
    sprites = [
        (factory.from_image("images/capybara_evil.png"), factory.from_image("images/capybara_evil_shadow.png"), Target.UGLY),
        (factory.from_image("images/capybara_good.png"), factory.from_image("images/capybara_silhouette.png"), Target.CUTE),
        (factory.from_image("images/cat_evil.png"), factory.from_image("images/cat_evil_shadow.png"), Target.UGLY),
        (factory.from_image("images/cat_good.png"), factory.from_image("images/cat_silhouette.png"), Target.CUTE),
        (factory.from_image("images/duck_evil.png"), factory.from_image("images/duck_evil_shadow.png"), Target.UGLY),
        (factory.from_image("images/duck_good.png"), factory.from_image("images/duck_silhouette.png"), Target.CUTE),
        (factory.from_image("images/Testing/Doge.png"), factory.from_image("images/Testing/Doge_inv.png"), Target.DANGER)
    ]
    for s1, s2, t in sprites:
        sdl2.SDL_SetTextureBlendMode(s1.texture, sdl2.SDL_BLENDMODE_BLEND)
        sdl2.SDL_SetTextureBlendMode(s2.texture, sdl2.SDL_BLENDMODE_BLEND)
    current_sprite, current_bg_sprite, current_target = random.choice(sprites)
    available_bg_positions: list = ([sdl2.SDL_Rect(200 * i, 200, 200, 200) for i in range(1, 9)] +
                                    [sdl2.SDL_Rect(200 * i, 400, 200, 200) for i in range(1, 9)] +
                                    [sdl2.SDL_Rect(200 * i, 600, 200, 200) for i in range(1, 9)])
    random.shuffle(available_bg_positions)
    sprite_museum: list = []
    for _, pos in sprite_museum:
        available_bg_positions.remove(pos)
    start_time: int = time_int()
    while running and trials < MAX_TRIALS:
        renderer.color = red
        renderer.clear()
        # if should_close():
        #     running = False
        #     break
        events = sdl2.ext.get_events()
        manual_choice = Action.NOTHING
        for event in events:
            if event.type == sdl2.SDL_QUIT or (event.type == sdl2.SDL_KEYUP and (event.key.keysym.sym == sdl2.SDLK_ESCAPE)):
                running = False
                break
            if event.key.keysym.sym == sdl2.SDLK_p:
                manual_choice = Action.PET
            if event.key.keysym.sym == sdl2.SDLK_a:
                manual_choice = Action.SLAP_LEFT
            if event.key.keysym.sym == sdl2.SDLK_d:
                manual_choice = Action.SLAP_RIGHT

        success, image = webcam.read()
        if not success:
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        sdl2.SDL_RenderClear(renderer.sdlrenderer)
        timestamp = time_int()
        if (timestamp - start_time) >= 1000 or game_mode != Mode.CONSEQUENCE or chosen_action == Action.NOTHING or current_target != Target.DANGER:
            sdl2.SDL_SetTextureAlphaMod(sprite_background.texture, 255)
            for sprite, position in sprite_museum:
                sdl2.SDL_SetTextureAlphaMod(sprite.texture, 255)
        else:
            sdl2.SDL_SetTextureAlphaMod(sprite_background.texture, int(255 * (.3 + abs(.3 - (timestamp - start_time) / 1000))))
            for sprite, position in sprite_museum:
                sdl2.SDL_SetTextureAlphaMod(sprite.texture, int(255 * (.3 + abs(.3 - (timestamp - start_time) / 1000))))
        sdl2.SDL_RenderCopy(renderer.sdlrenderer, sprite_background.texture, None, background_rect)
        percentage: float = (timestamp - start_time) / 1000
        for i in range(len(sprite_museum)):
            if (game_mode == Mode.CONSEQUENCE and
                    ((current_target == Target.CUTE and chosen_action == Action.PET) or (current_target == Target.DANGER and chosen_action == Action.NOTHING)) and
                    percentage < 1):
                current_rect = sdl2.SDL_Rect(sprite_museum[i][1].x, sprite_museum[i][1].y - int(60 * percentage * (1 - percentage)),
                                             sprite_museum[i][1].w, sprite_museum[i][1].h)
                sdl2.SDL_RenderCopy(renderer.sdlrenderer, sprite_museum[i][0].texture, None, current_rect)
            else:
                sdl2.SDL_RenderCopy(renderer.sdlrenderer, sprite_museum[i][0].texture, None, sprite_museum[i][1])
        sdl2.SDL_UpdateTexture(webcam_texture, None, image.ctypes.data, width * 3)
        sdl2.SDL_RenderCopy(renderer.sdlrenderer, webcam_texture, None, webcam_rect)
        if game_mode == Mode.LOAD:
            percentage: float = (timestamp - start_time) / 500
            if percentage >= 1:
                start_time = timestamp
                game_mode = Mode.CHOICE
                time_displayed = CHOICE_SECONDS
                sdl2.SDL_RenderCopy(renderer.sdlrenderer, current_sprite.texture, None, image_rect)
            else:
                current_step_rect = sdl2.SDL_Rect(image_rect.x + int(image_rect.w * (1 - percentage) / 2),
                                                  image_rect.y + int(image_rect.h * (1 - percentage)),
                                                  int(image_rect.w * percentage), int(image_rect.h * percentage))
                sdl2.SDL_RenderCopy(renderer.sdlrenderer, current_sprite.texture, None, current_step_rect)
        elif game_mode == Mode.CHOICE:
            sdl2.SDL_RenderCopy(renderer.sdlrenderer, current_sprite.texture, None, image_rect)
            sdl2.SDL_RenderCopy(renderer.sdlrenderer, timer_textures[time_displayed - 1], None, timer_rect)
            if timestamp - start_time >= 1000:
                start_time = timestamp
                time_displayed -= 1
                if time_displayed == 0:
                    game_mode = Mode.CONSEQUENCE
                    start_time = timestamp
            if manual_choice != Action.NOTHING:
                chosen_action = manual_choice
                game_mode = Mode.CONSEQUENCE
                start_time = timestamp
            hand_landmarks = slap_checker.detect_hand(image)
            if hand_landmarks is not None:
                for landmark in hand_landmarks:
                    x = int(landmark.x * 300) - 2
                    y = int(landmark.y * 200) - 2
                    sdl2.SDL_SetRenderDrawColor(renderer.sdlrenderer, 0, 0, 255, 255)
                    sdl2.SDL_RenderFillRect(renderer.sdlrenderer, sdl2.SDL_Rect(webcam_rect.x + x, webcam_rect.y + y, 4, 4))
                for start, end in [
                    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5),
                    (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17)
                ]:
                    sdl2.SDL_SetRenderDrawColor(renderer.sdlrenderer, 0, 255, 0, 255)
                    sdl2.SDL_RenderDrawLine(renderer.sdlrenderer,
                        webcam_rect.x + int(hand_landmarks[start].x * 300), webcam_rect.y + int(hand_landmarks[start].y * 200),
                        webcam_rect.x + int(hand_landmarks[end].x * 300), webcam_rect.y + int(hand_landmarks[end].y * 200))
                action = slap_checker.check_slap(hand_landmarks)
                if action != Action.NOTHING:
                    chosen_action = action
                    game_mode = Mode.CONSEQUENCE
                    slap_checker.reset()
                    start_time = timestamp

        elif game_mode == Mode.CONSEQUENCE:
            sdl2.SDL_RenderCopy(renderer.sdlrenderer, actions_textures[chosen_action.value], None, sdl2.SDL_Rect(50, 50, 1870, 100))
            if timestamp - start_time >= 1000:
                if (current_target == Target.CUTE and chosen_action == Action.PET) or (current_target == Target.DANGER and chosen_action == Action.NOTHING):
                    chosen_position = available_bg_positions.pop(-1)
                    sprite_museum.append((current_bg_sprite, chosen_position))
                    sdl2.SDL_SetTextureAlphaMod(current_bg_sprite.texture, 255)
                    sdl2.SDL_RenderCopy(renderer.sdlrenderer, current_bg_sprite.texture, None, chosen_position)
                    random.shuffle(sprite_museum)
                if current_target == Target.UGLY and chosen_action != Action.SLAP_RIGHT and chosen_action != Action.SLAP_LEFT and len(sprite_museum) > 0:
                    available_bg_positions.append(sprite_museum.pop(len(sprite_museum) - 1)[1])
                    random.shuffle(available_bg_positions)
                start_time = timestamp
                current_sprite, current_bg_sprite, current_target = random.choice(sprites)
                sdl2.SDL_SetTextureAlphaMod(current_sprite.texture, 255)
                game_mode = Mode.LOAD
                chosen_action = Action.NOTHING
                trials += 1
            else:
                percentage: float = (timestamp - start_time) / 1000
                if current_target == Target.UGLY and (chosen_action == Action.NOTHING or chosen_action == Action.PET) and len(sprite_museum) > 0:
                    sdl2.SDL_SetTextureAlphaMod(current_bg_sprite.texture, int(255 * percentage))
                    sdl2.SDL_RenderCopy(renderer.sdlrenderer, current_bg_sprite.texture, None, sprite_museum[-1][1])
                elif chosen_action == Action.NOTHING and current_target != Target.DANGER:
                    sdl2.SDL_SetTextureAlphaMod(current_sprite.texture, 255 - int(255 * percentage))
                    sdl2.SDL_RenderCopy(renderer.sdlrenderer, current_sprite.texture, None, image_rect)
                elif (chosen_action == Action.PET and current_target == Target.CUTE) or (chosen_action == Action.NOTHING and current_target == Target.DANGER):
                    current_rect = sdl2.SDL_Rect(
                        int(available_bg_positions[-1][0] * percentage + image_rect.x * (1 - percentage)),
                        int(available_bg_positions[-1][1] * percentage + image_rect.y * (1 - percentage)),
                        int(image_rect.w + percentage * (200 - image_rect.w)),
                        int(image_rect.h + percentage * (200 - image_rect.h))
                    )
                    sdl2.SDL_SetTextureAlphaMod(current_sprite.texture, 255 - int(255 * percentage))
                    sdl2.SDL_SetTextureAlphaMod(current_bg_sprite.texture, int(255 * percentage))
                    sdl2.SDL_RenderCopy(renderer.sdlrenderer, current_sprite.texture, None, current_rect)
                    sdl2.SDL_RenderCopy(renderer.sdlrenderer, current_bg_sprite.texture, None, current_rect)
                elif chosen_action == Action.SLAP_LEFT:
                    current_rect = sdl2.SDL_Rect(
                        int((-SCREEN_W / 3 - image_rect.w / 2) * percentage + image_rect.x * (1 - percentage)),
                        int(SCREEN_H / 3 * percentage + image_rect.y * (1 - percentage)),
                        image_rect.w,
                        image_rect.h
                    )
                    sdl2.SDL_RenderCopyEx(renderer.sdlrenderer, current_sprite.texture, None, current_rect, -1080 * percentage, None, sdl2.SDL_FLIP_NONE)
                elif chosen_action == Action.SLAP_RIGHT:
                    current_rect = sdl2.SDL_Rect(
                        int((SCREEN_W * 4 / 3 - image_rect.w / 2) * percentage + image_rect.x * (1 - percentage)),
                        int(SCREEN_H / 3 * percentage + image_rect.y * (1 - percentage)),
                        image_rect.w,
                        image_rect.h
                    )
                    sdl2.SDL_RenderCopyEx(renderer.sdlrenderer, current_sprite.texture, None, current_rect, 1080 * percentage, None, sdl2.SDL_FLIP_NONE)
        else:
            print("The fuck?")
            break
        current_tick: int = time_int()
        if (current_tick - prev_tick) * FRAMERATE < 1000:
            time.sleep((1000 / FRAMERATE - current_tick + prev_tick) / 1000)
        renderer.present()
        prev_tick = time_int()
    if running:
        renderer.color = black
        for i in range(2 * FRAMERATE, 0, -1):
            if should_close():
                running = False
                break
            sdl2.SDL_RenderClear(renderer.sdlrenderer)
            sdl2.SDL_SetTextureAlphaMod(sprite_background.texture, int(255 * i / (2 * FRAMERATE)))
            sdl2.SDL_RenderCopy(renderer.sdlrenderer, sprite_background.texture, None, background_rect)
            for sprite, position in sprite_museum:
                sdl2.SDL_SetTextureAlphaMod(sprite.texture, int(255 * i / (2 * FRAMERATE)))
                sdl2.SDL_RenderCopy(renderer.sdlrenderer, sprite.texture, None, position)
            current_tick: int = time_int()
            if (current_tick - prev_tick) * FRAMERATE < 1000:
                time.sleep((1000 / FRAMERATE - current_tick + prev_tick) / 1000)
            renderer.present()
            prev_tick = time_int()
    sdl2.SDL_SetTextureAlphaMod(sprite_background.texture, 255)
    for sprite, position in sprite_museum:
        sdl2.SDL_SetTextureAlphaMod(sprite.texture, 255)
    jump_x: list[int] = [0 for _ in sprite_museum]
    text_surface = sdl2.sdlttf.TTF_RenderUTF8_Blended(font, f"FINAL SCORE: {len(sprite_museum)} cheerers!".encode("utf-8"), white)
    result_texture = sdl2.SDL_CreateTextureFromSurface(renderer.sdlrenderer, text_surface)
    result_rect = sdl2.SDL_Rect(50, 50, 1820, 100)
    sdl2.SDL_FreeSurface(text_surface)
    time.sleep(.5)
    if len(sprite_museum) == 0:
        evil_sprites = list(map(lambda t: t[1], filter(lambda s: s[2] == Target.UGLY, sprites)))
        sprite_museum = [(random.choice(evil_sprites), pos) for pos in available_bg_positions]
    sprite_museum.sort(key=lambda t: (t[1][1], t[1][0]))
    while running:
        if should_close():
            running = False
            break
        sdl2.SDL_RenderClear(renderer.sdlrenderer)
        sdl2.SDL_RenderCopy(renderer.sdlrenderer, sprite_background.texture, None, background_rect)
        sdl2.SDL_RenderCopy(renderer.sdlrenderer, result_texture, None, result_rect)
        for i in range(len(jump_x)):
            if jump_x[i] == 0 and random.randint(1, 100) == 60:
                jump_x[i] = 30
            current_rect = sdl2.SDL_Rect(sprite_museum[i][1].x, sprite_museum[i][1].y - 10 * int(jump_x[i] * (1 - jump_x[i] / 30)),
                                         sprite_museum[i][1].w, sprite_museum[i][1].h)
            if jump_x[i] > 0:
                jump_x[i] -= 1
            sdl2.SDL_RenderCopy(renderer.sdlrenderer, sprite_museum[i][0].texture, None, current_rect)
        current_tick: int = time_int()
        if (current_tick - prev_tick) * FRAMERATE < 1000:
            time.sleep((1000 / FRAMERATE - current_tick + prev_tick) / 1000)
        renderer.present()
        prev_tick = time_int()
    slap_checker.shutdown()
    sdl2.sdlttf.TTF_CloseFont(font)
    sdl2.sdlttf.TTF_Quit()
    sdl2.SDL_Quit()
