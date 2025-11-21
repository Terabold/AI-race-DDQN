import pygame
import math
from scripts.Constants import *
from pathlib import Path


def font_scale(size, Font=FONT):
    return pygame.font.Font(Font, size)


def create_shadowed_text(text, font, color, shadow_color=BLACK, offset=4):
    shadow = font.render(text, True, shadow_color)
    main_text = font.render(text, True, color)
    combined = pygame.Surface((shadow.get_width() + offset, shadow.get_height() + offset), pygame.SRCALPHA)
    combined.blit(shadow, (offset, offset))
    combined.blit(main_text, (0, 0))
    return combined


def smooth_sine_wave(time, period=4.0, min_val=0.0, max_val=1.0):
    normalized = (math.cos(time * (2 * math.pi / period)) + 1) / 2
    return min_val + normalized * (max_val - min_val)


def load_sound(path, volume=DEFAULT_SOUND_VOLUME):
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    sound = pygame.mixer.Sound(str(Path(path)))
    sound.set_volume(volume)
    return sound


def calculate_ui_constants(display_size):
    ref_width, ref_height = 1920, 1080
    width_scale = display_size[0] / ref_width
    height_scale = display_size[1] / ref_height
    general_scale = min(width_scale, height_scale)
    
    return {
        'BUTTON_HEIGHT': int(80 * height_scale),
        'BUTTON_MIN_WIDTH': int(200 * width_scale),
        'BUTTON_TEXT_PADDING': int(40 * general_scale),
        'BUTTON_SPACING': int(20 * general_scale),
        'BUTTON_COLOR': (40, 40, 70, 220),
        'BUTTON_HOVER_COLOR': (60, 60, 100, 240),
    }


# ============================================================================
# GAME OVERLAYS
# ============================================================================

def _draw_game_overlay(environment, title, title_color, overlay_tint=None):
    """Helper to draw game end overlays"""
    current_time = pygame.time.get_ticks() / 1000
    
    # Overlay
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))
    if overlay_tint:
        pygame.draw.rect(overlay, overlay_tint, (0, 0, WIDTH, HEIGHT))
    environment.surface.blit(overlay, (0, 0))
    
    # Title
    title_text = create_shadowed_text(title, font_scale(80, FONT), title_color, BLACK, 5)
    environment.surface.blit(title_text, title_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 120)))
    
    # Player statuses
    y = HEIGHT//2 - 20
    for player_num, (active, finished, car) in enumerate([
        (environment.car1_active, environment.car1_finished, environment.car1),
        (environment.car2_active, environment.car2_finished, environment.car2)
    ], 1):
        if active:
            if title == "Race Failed!":
                status = "Crashed!" if car.failed else "Time Up!"
            else:
                status = "Finished!" if finished else "Time Up!"
            
            color = DODGERBLUE if player_num == 1 else RED
            text = create_shadowed_text(f"Player {player_num}: {status}", font_scale(42, FONT), color)
            environment.surface.blit(text, text.get_rect(center=(WIDTH//2, y)))
            y += 60
    
    # Pulsing prompt
    prompt = "Press SPACE to " + ("try again" if title == "Race Failed!" else "restart")
    period = 1.8 if title == "Race Failed!" else 1.2
    restart_text = font_scale(36, FONT).render(prompt, True, WHITE)
    restart_text.set_alpha(int(255 * smooth_sine_wave(current_time, period=period, min_val=0.0 if title == "Race Finished!" else 0.1, max_val=1.0)))
    environment.surface.blit(restart_text, restart_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 140)))


def draw_finished(environment):
    _draw_game_overlay(environment, "Race Finished!", GREEN)


def draw_failed(environment):
    _draw_game_overlay(environment, "Race Failed!", RED, overlay_tint=(255, 0, 0, 30))


def draw_ui(environment):
    """Draw game UI (timers, status)"""
    y = 10
    
    for player_num, (active, finished, car, time) in enumerate([
        (environment.car1_active, environment.car1_finished, environment.car1, environment.car1_time),
        (environment.car2_active, environment.car2_finished, environment.car2, environment.car2_time)
    ], 1):
        if active:
            if car.failed:
                status, color = "Failed!", RED
            elif finished:
                status, color = f"P{player_num}: Finished!", GREEN
            else:
                status, color = f"P{player_num} Time: {time:.1f}", RED if time < 3 else GREEN
            
            timer_text = create_shadowed_text(status, font_scale(32, FONT), color)
            environment.surface.blit(timer_text, (15, y))
            y += 40


def draw_countdown(environment, count):
    # Shadow
    shadow = font_scale(180, COUNTDOWN_FONT).render(str(count), True, BLACK)
    shadow_surface = pygame.Surface(shadow.get_size(), pygame.SRCALPHA)
    shadow_surface.blit(shadow, (0, 0))
    shadow_surface.set_alpha(200)
    environment.surface.blit(shadow_surface, shadow_surface.get_rect(center=(WIDTH // 2 + 6, HEIGHT // 2 + 6)))
    
    # Main text
    text = font_scale(180, COUNTDOWN_FONT).render(str(count), True, RED)
    environment.surface.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2)))


# ============================================================================
# MENU CLASSES
# ============================================================================

class Button:
    def __init__(self, rect, text, action, font, menu, bg_color=None):
        self.rect = rect
        self.text = text
        self.action = action
        self.font = font
        self.menu = menu
        self.selected = False
        self.bg_color = bg_color
        self.border_radius = max(6, int(rect.height * 0.1))

    def update_hover_state(self, mouse_pos):
        self.selected = self.rect.collidepoint(mouse_pos)

    def draw(self, surface):
        color = self.bg_color or (70, 70, 70)
        if self.selected:
            color = tuple(min(c + 30, 255) for c in color) if self.bg_color else (100, 100, 100)
        
        pygame.draw.rect(surface, color, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2, border_radius=self.border_radius)
        
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        surface.blit(text_surf, text_surf.get_rect(center=self.rect.center))


class MenuScreen:
    def __init__(self, menu, title="Menu"):
        self.menu = menu
        self.screen = menu.screen
        self.UI_CONSTANTS = calculate_ui_constants(DISPLAY_SIZE)
        self.font = pygame.font.Font(FONT, 40)
        self.title_font = pygame.font.Font(FONT, 70)
        self.enabled = False
        self.title = title
        self.buttons = []

    def enable(self):
        self.enabled = True
        self.initialize()

    def disable(self):
        self.enabled = False

    def initialize(self):
        pass

    def update(self, events):
        if not self.enabled:
            return
        
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.update_hover_state(mouse_pos)
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in self.buttons:
                    if button.selected:
                        button.action()
                        return

    def draw(self, surface):
        if not self.enabled:
            return
        
        title_text = self.title_font.render(self.title, True, (255, 255, 255))
        surface.blit(title_text, ((surface.get_width() - title_text.get_width()) // 2, int(surface.get_height() * 0.1)))
        
        for button in self.buttons:
            button.draw(surface)

    def clear_buttons(self):
        self.buttons.clear()

    def create_button(self, text, action, x, y, width=None, bg_color=None):
        if width is None:
            text_surf = self.font.render(text, True, (255, 255, 255))
            width = max(text_surf.get_width() + self.UI_CONSTANTS['BUTTON_TEXT_PADDING'], self.UI_CONSTANTS['BUTTON_MIN_WIDTH'])
        
        button = Button(pygame.Rect(x, y, width, self.UI_CONSTANTS['BUTTON_HEIGHT']), text, action, self.font, self.menu, bg_color)
        self.buttons.append(button)
        return button