import pygame
import math
from scripts.Constants import FONT, COUNTDOWN_FONT, WIDTH, HEIGHT, BLACK, WHITE, RED, GREEN, DODGERBLUE
from scripts.ResourceLoader import resource_manager


def font_scale(size, Font=FONT):
    return resource_manager.get_font(Font, size)


def create_shadowed_text(text, font, color, shadow_color=BLACK, offset=4):
    shadow = font.render(text, True, shadow_color)
    main_text = font.render(text, True, color)
    # יצירת משטח חדש עם תמיכה בשקיפות
    combined = pygame.Surface((shadow.get_width() + offset, shadow.get_height() + offset), pygame.SRCALPHA)
    combined.blit(shadow, (offset, offset)) # ציור הצל בהסטה
    combined.blit(main_text, (0, 0)) # ציור הטקסט הראשי
    return combined


def smooth_sine_wave(time, period=4.0, min_val=0.0, max_val=1.0):
    # חישוב ערך גל סינוסי מנורמל לטווח 0-1 (ליצירת אפקט פעימה חלק)
    normalized = (math.cos(time * (2 * math.pi / period)) + 1) / 2
    return min_val + normalized * (max_val - min_val)


def calculate_ui_constants(display_size):
    # חישוב קבועים מותאמי רזולוציה לפי מסך ייחוס (1920x1080)
    ref_width, ref_height = 1920, 1080
    width_scale = display_size[0] / ref_width
    height_scale = display_size[1] / ref_height
    # לוקחים את המינימום כדי לשמור על פרופורציות בלי חריגה מהמסך
    general_scale = min(width_scale, height_scale)
    
    return {
        'BUTTON_HEIGHT': int(80 * height_scale),
        'BUTTON_MIN_WIDTH': int(200 * width_scale),
        'BUTTON_TEXT_PADDING': int(40 * general_scale),
        'BUTTON_SPACING': int(20 * general_scale),
    }


# שכבות-על לסיום משחק

def draw_game_overlay(environment, title, title_color, overlay_tint=None):
    current_time = pygame.time.get_ticks() / 1000
    
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA) # יצירת מסך שקוף למחצה
    overlay.fill((0, 0, 0, 128)) # מילוי המסך בצבע שחור חלקי שקיפות
    if overlay_tint:
        pygame.draw.rect(overlay, overlay_tint, (0, 0, WIDTH, HEIGHT)) # אם יש צבע נוסף, מוסיף אותו לשקיפות
    environment.surface.blit(overlay, (0, 0)) # ציור המסך על הרקע
    
    title_text = create_shadowed_text(title, font_scale(80, FONT), title_color, BLACK, 5)
    environment.surface.blit(title_text, title_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 120)))
    
    y = HEIGHT//2 - 20
    car1 = environment.car1
    car2 = environment.car2
    
    for player_num, (active, finished, car) in enumerate([
        (environment.car1_active, environment.car1_finished, car1),
        (environment.car2_active, environment.car2_finished, car2)
    ], 1):
        if active and car is not None:
            # זיהוי מצב כל שחקן בנפרד
            if car.failed:
                status = "Crashed!"
            elif finished:
                status = "Finished!"
            else:
                status = "Time Up!"
            
            color = DODGERBLUE if player_num == 1 else RED
            text = create_shadowed_text(f"Player {player_num}: {status}", font_scale(42, FONT), color)
            environment.surface.blit(text, text.get_rect(center=(WIDTH//2, y)))
            y += 60
    
    prompt = "Press SPACE to " + ("try again" if title == "Race Failed!" else "restart")
    period = 1.8 if title == "Race Failed!" else 1.2
    restart_text = font_scale(36, FONT).render(prompt, True, WHITE)
    restart_text.set_alpha(int(255 * smooth_sine_wave(current_time, period=period, min_val=0.0 if title == "Race Finished!" else 0.1, max_val=1.0)))
    environment.surface.blit(restart_text, restart_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 140)))


def draw_finished(environment):
    draw_game_overlay(environment, "Race Finished!", GREEN)


def draw_failed(environment):
    draw_game_overlay(environment, "Race Failed!", RED, overlay_tint=(255, 0, 0, 30))


def draw_ui(environment):
    # ציור טיימר ומצב רכב במהלך משחק
    y = 10
    car1 = environment.car1
    car2 = environment.car2
    
    for player_num, (active, finished, car, time) in enumerate([
        (environment.car1_active, environment.car1_finished, car1, environment.car1_time),
        (environment.car2_active, environment.car2_finished, car2, environment.car2_time)
    ], 1):
        if active and car is not None:
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
    # ציור ספירת הזינוק האחורה
    shadow = font_scale(180, COUNTDOWN_FONT).render(str(count), True, BLACK)
    shadow_surface = pygame.Surface(shadow.get_size(), pygame.SRCALPHA)
    shadow_surface.blit(shadow, (0, 0))
    shadow_surface.set_alpha(200)
    environment.surface.blit(shadow_surface, shadow_surface.get_rect(center=(WIDTH // 2 + 6, HEIGHT // 2 + 6)))
    
    text = font_scale(180, COUNTDOWN_FONT).render(str(count), True, RED)
    environment.surface.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2)))


class Button:
    def __init__(self, rect, text, action, font, bg_color=None):
        self.rect = rect
        self.text = text
        self.action = action
        self.font = font
        self.selected = False
        self.bg_color = bg_color
        self.border_radius = max(6, int(rect.height * 0.1))

    def update_hover_state(self, mouse_pos):
        self.selected = self.rect.collidepoint(mouse_pos)