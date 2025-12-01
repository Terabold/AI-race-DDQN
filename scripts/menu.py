# all the menus
import pygame
import sys
from scripts.Constants import FONT, WHITE, BLACK, COLORS, CAR_COLORS, CAR_COLORS_LIST, GOLD, WIDTH, HEIGHT
from scripts.utils import Button, calculate_ui_constants
from scripts.GameManager import game_state_manager
from pathlib import Path


# base class

class BaseMenuScreen:
    def __init__(self, screen, title="Menu"):
        self.screen = screen
        self.title = title
        self.UI = calculate_ui_constants((WIDTH, HEIGHT))
        self.font = pygame.font.Font(FONT, 40)
        self.title_font = pygame.font.Font(FONT, 70)
        self.buttons = []
        self.initialize()
    
    def initialize(self):
        pass
    
    def create_button(self, text, action, x, y, width=None, bg_color=None, image=None):
        if width is None:
            text_surf = self.font.render(text, True, WHITE)
            width = max(text_surf.get_width() + self.UI['BUTTON_TEXT_PADDING'], 
                       self.UI['BUTTON_MIN_WIDTH'])
        
        btn = Button(
            pygame.Rect(x, y, width, self.UI['BUTTON_HEIGHT']),
            text, action, self.font, bg_color
        )
        btn.image = image
        btn.disabled = False
        self.buttons.append(btn)
        return btn
    
    def draw_title(self):
        title = self.title_font.render(self.title, True, COLORS["title"])
        shadow = self.title_font.render(self.title, True, BLACK)
        cx = (self.screen.get_width() - title.get_width()) // 2
        ty = int(self.screen.get_height() * 0.05)
        self.screen.blit(shadow, (cx + 4, ty + 4))
        self.screen.blit(title, (cx, ty))
    
    def draw_button(self, btn, selected=False, highlight_color=None):
        if btn.disabled:
            bg = (40, 40, 40)
        elif selected:
            bg = highlight_color or (100, 150, 255)
        else:
            bg = btn.bg_color or (70, 70, 70)
        
        if btn.selected and not btn.disabled:
            bg = tuple(min(c + 30, 255) for c in bg)
        
        pygame.draw.rect(self.screen, bg, btn.rect, border_radius=btn.border_radius)
        
        border_color = WHITE if selected and not btn.disabled else (200, 200, 200)
        border_width = 3 if selected else 2
        pygame.draw.rect(self.screen, border_color, btn.rect, border_width, 
                        border_radius=btn.border_radius)
        
        if btn.image:
            img_rect = btn.image.get_rect(center=btn.rect.center)
            self.screen.blit(btn.image, img_rect)
        elif btn.text:
            text_color = (100, 100, 100) if btn.disabled else WHITE
            text_surf = btn.font.render(btn.text, True, text_color)
            self.screen.blit(text_surf, text_surf.get_rect(center=btn.rect.center))
        
        if btn.disabled:
            pad = 8
            x1, y1 = btn.rect.left + pad, btn.rect.top + pad
            x2, y2 = btn.rect.right - pad, btn.rect.bottom - pad
            pygame.draw.line(self.screen, (200, 50, 50), (x1, y1), (x2, y2), 3)
            pygame.draw.line(self.screen, (200, 50, 50), (x1, y2), (x2, y1), 3)
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.on_escape()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for btn in self.buttons:
                    if btn.selected and not btn.disabled:
                        btn.action()
                        return
    
    def on_escape(self):
        pass
    
    def update(self):
        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons:
            btn.update_hover_state(mouse_pos)
    
    def draw(self):
        self.draw_title()
        for btn in self.buttons:
            self.draw_button(btn)
    
    def run(self):
        self.handle_events()
        self.update()
        self.draw()


# main menu

class MainMenu(BaseMenuScreen):
    def __init__(self, screen):
        super().__init__(screen, "RACING GAME")
    
    def initialize(self):
        cx = self.screen.get_width() // 2
        start_y = int(self.screen.get_height() * 0.3)
        width = int(self.screen.get_width() * 0.25)
        spacing = self.UI['BUTTON_HEIGHT'] + self.UI['BUTTON_SPACING']
        
        buttons = [
            ('PLAY', lambda: game_state_manager.setState('settings'), None),
            ('TRAIN AI', lambda: game_state_manager.setState('training'), (70, 100, 180)),
            ('TEST AI', lambda: game_state_manager.setState('tester_settings'), (180, 100, 70)),
            ('QUIT', self._quit, (200, 50, 50))
        ]
        
        for i, (text, action, color) in enumerate(buttons):
            self.create_button(text, action, cx - width // 2, start_y + i * spacing, width, color)
    
    def _quit(self):
        pygame.quit()
        sys.exit()


# race settings

class RaceSettingsMenu(BaseMenuScreen):
    def __init__(self, screen):
        self.car_images = {}
        self.p1_type_btns = []
        self.p2_type_btns = []
        self.p1_car_btns = []
        self.p2_car_btns = []
        self.info_font = pygame.font.Font(FONT, int(screen.get_height() * 0.02))
        super().__init__(screen, "Race Settings")
    
    def load_car_image(self, color):
        if color not in self.car_images:
            path = Path(CAR_COLORS[color])
            if path.exists():
                img = pygame.image.load(path)
                img = pygame.transform.rotate(img, 90)
                self.car_images[color] = pygame.transform.scale(img, (100, 50))
        return self.car_images.get(color)
    
    def initialize(self):
        self.buttons.clear()
        self.p1_type_btns.clear()
        self.p2_type_btns.clear()
        self.p1_car_btns.clear()
        self.p2_car_btns.clear()
        
        w, h = self.screen.get_size()
        cx = w // 2
        col_offset = int(w * 0.12)
        car_offset = int(w * 0.16)
        
        p1_x, p2_x = cx - col_offset, cx + col_offset
        p1_car_x = cx - car_offset - col_offset
        p2_car_x = cx + car_offset + col_offset
        
        btn_width = int(w * 0.15)
        car_width = int(w * 0.1)
        
        top = int(h * 0.20)
        spacing = self.UI['BUTTON_HEIGHT'] + int(self.UI['BUTTON_SPACING'] * 0.8)
        
        # player type buttons
        for i, ptype in enumerate(["Human", "DQN"]):
            y = top + (i + 1) * spacing
            b1 = self.create_button(ptype, lambda pt=ptype: self.toggle_p1(pt), 
                                   p1_x - btn_width//2, y, btn_width)
            b2 = self.create_button(ptype, lambda pt=ptype: self.toggle_p2(pt), 
                                   p2_x - btn_width//2, y, btn_width)
            self.p1_type_btns.append(b1)
            self.p2_type_btns.append(b2)
        
        # car buttons
        for i, color in enumerate(CAR_COLORS_LIST):
            y = top + (i + 1) * spacing
            img = self.load_car_image(color)
            b1 = self.create_button("", lambda c=color: self.select_p1_car(c), 
                                   p1_car_x - car_width//2, y, car_width, image=img)
            b2 = self.create_button("", lambda c=color: self.select_p2_car(c), 
                                   p2_car_x - car_width//2, y, car_width, image=img)
            b1.color_name = color
            b2.color_name = color
            self.p1_car_btns.append(b1)
            self.p2_car_btns.append(b2)
        
        self.create_button("Start", self.start, cx - 150, int(h * 0.85), 300, COLORS["start"])
        self.create_button("←", lambda: game_state_manager.setState('menu'), 
                          int(w * 0.02), int(h * 0.02), int(w * 0.08))
    
    def toggle_p1(self, ptype):
        current = game_state_manager.player1_selection
        game_state_manager.player1_selection = None if current == ptype else ptype
    
    def toggle_p2(self, ptype):
        current = game_state_manager.player2_selection
        game_state_manager.player2_selection = None if current == ptype else ptype
    
    def select_p1_car(self, color):
        if color != game_state_manager.player2_car_color:
            game_state_manager.player1_car_color = color
    
    def select_p2_car(self, color):
        if color != game_state_manager.player1_car_color:
            game_state_manager.player2_car_color = color
    
    def start(self):
        if game_state_manager.player1_selection or game_state_manager.player2_selection:
            game_state_manager.setState('game')
    
    def on_escape(self):
        game_state_manager.setState('menu')
    
    def update(self):
        super().update()
        for btn in self.p1_car_btns:
            btn.disabled = btn.color_name == game_state_manager.player2_car_color
        for btn in self.p2_car_btns:
            btn.disabled = btn.color_name == game_state_manager.player1_car_color
    
    def draw(self):
        self.draw_title()
        self.draw_labels()
        
        ptypes = ["Human", "DQN"]
        for i, btn in enumerate(self.p1_type_btns):
            selected = game_state_manager.player1_selection == ptypes[i]
            self.draw_button(btn, selected, COLORS["p1"])
        for i, btn in enumerate(self.p2_type_btns):
            selected = game_state_manager.player2_selection == ptypes[i]
            self.draw_button(btn, selected, COLORS["p2"])
        
        for btn in self.p1_car_btns:
            selected = btn.color_name == game_state_manager.player1_car_color
            self.draw_button(btn, selected, COLORS["p1"])
        for btn in self.p2_car_btns:
            selected = btn.color_name == game_state_manager.player2_car_color
            self.draw_button(btn, selected, COLORS["p2"])
        
        for btn in self.buttons[-2:]:
            self.draw_button(btn)
        
        self.draw_controls()
    
    def draw_labels(self):
        w = self.screen.get_width()
        cx = w // 2
        col_offset = int(w * 0.12)
        car_offset = int(w * 0.16)
        y = int(self.screen.get_height() * 0.22)
        
        labels = [
            ("Player1", cx - col_offset, COLORS["p1"]),
            ("Player2", cx + col_offset, COLORS["p2"]),
            ("Car", cx - car_offset - col_offset, COLORS["p1"]),
            ("Car", cx + car_offset + col_offset, COLORS["p2"])
        ]
        
        for text, x, color in labels:
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, surf.get_rect(center=(x, y)))
    
    def draw_controls(self):
        w, h = self.screen.get_size()
        y = int(h * 0.5)
        
        if game_state_manager.player1_selection:
            self.draw_control_panel(int(w * 0.08), y, True)
        if game_state_manager.player2_selection:
            self.draw_control_panel(int(w * 0.92), y, False)
    
    def draw_control_panel(self, x, y, is_p1):
        ctrl = ({'Forward': 'W', 'Backward': 'S', 'Left': 'A', 'Right': 'D'} if is_p1 
                else {'Forward': 'Up', 'Backward': 'Down', 'Left': 'Left', 'Right': 'Right'})
        color = COLORS["p1"] if is_p1 else COLORS["p2"]
        
        box = pygame.Rect(x - 100, y - 125, 200, 250)
        
        for i in range(3):
            bg = pygame.Surface((200 - i*2, 250 - i*2))
            bg.set_alpha(100 - i*20)
            bg.fill(color)
            self.screen.blit(bg, (box.x + i, box.y + i))
        
        pygame.draw.rect(self.screen, COLORS["border"], box, 3)
        
        title = self.info_font.render("Controls", True, COLORS["button_bg"])
        self.screen.blit(title, title.get_rect(center=(x, box.y + 25)))
        
        small_font = pygame.font.Font(FONT, 12)
        for i, (action, key) in enumerate(ctrl.items()):
            ky = box.y + 80 + i * 45
            
            action_text = small_font.render(action, True, COLORS["button_bg"])
            self.screen.blit(action_text, action_text.get_rect(center=(x - 45, ky)))
            
            key_rect = pygame.Rect(x + 15, ky - 17, 60, 35)
            key_bg = pygame.Surface((60, 35))
            key_bg.set_alpha(160)
            key_bg.fill(color)
            self.screen.blit(key_bg, key_rect)
            pygame.draw.rect(self.screen, COLORS["border"], key_rect, 2)
            
            key_text = small_font.render(key, True, COLORS["button_bg"])
            self.screen.blit(key_text, key_text.get_rect(center=(x + 45, ky)))


# tester settings

class TesterSettingsMenu(BaseMenuScreen):
    def __init__(self, screen):
        self.num_cars = 10
        self.preset_btns = []
        self.header_font = pygame.font.Font(FONT, int(screen.get_height() * 0.025))
        self.info_font = pygame.font.Font(FONT, int(screen.get_height() * 0.02))
        self.count_font = pygame.font.Font(FONT, int(screen.get_height() * 0.08))
        super().__init__(screen, "AI Performance Test")
    
    def initialize(self):
        self.buttons.clear()
        self.preset_btns.clear()
        
        w, h = self.screen.get_size()
        cx = w // 2
        btn_width = int(w * 0.12)
        
        presets = [1, 5, 10, 25, 50, 100]
        preset_y = int(h * 0.35)
        total_width = len(presets) * btn_width + (len(presets) - 1) * 20
        start_x = cx - total_width // 2
        
        for i, num in enumerate(presets):
            x = start_x + i * (btn_width + 20)
            btn = self.create_button(str(num), lambda n=num: self.set_count(n),
                                    x, preset_y, btn_width, (70, 100, 180))
            btn.preset_value = num
            self.preset_btns.append(btn)
        
        self.create_button("Test", self.start_test, cx - 150, int(h * 0.85), 300, (70, 180, 70))
        self.create_button("←", lambda: game_state_manager.setState('menu'),
                          int(w * 0.02), int(h * 0.02), int(w * 0.08))
    
    def set_count(self, num):
        self.num_cars = num
    
    def start_test(self):
        game_state_manager.tester_num_cars = self.num_cars
        game_state_manager.setState('tester')
    
    def on_escape(self):
        game_state_manager.setState('menu')
    
    def draw(self):
        self.draw_title()
        
        w, h = self.screen.get_size()
        
        for i, line in enumerate([
            "Select number of AI cars to test performance",
            "All cars will race simultaneously using trained AI model",
        ]):
            text = self.info_font.render(line, True, (200, 200, 200))
            self.screen.blit(text, text.get_rect(center=(w//2, int(h * 0.18) + i * 30)))
        
        label = self.header_font.render("Preset Amounts", True, WHITE)
        self.screen.blit(label, label.get_rect(center=(w//2, int(h * 0.28))))
        
        count_y = int(h * 0.55)
        count_text = f"{self.num_cars} Cars"
        count_shadow = self.count_font.render(count_text, True, BLACK)
        count_surf = self.count_font.render(count_text, True, GOLD)
        self.screen.blit(count_shadow, count_shadow.get_rect(center=(w//2 + 3, count_y + 3)))
        self.screen.blit(count_surf, count_surf.get_rect(center=(w//2, count_y)))
        
        for btn in self.preset_btns:
            selected = self.num_cars == btn.preset_value
            self.draw_button(btn, selected)
        
        for btn in self.buttons[-2:]:
            self.draw_button(btn)


# pause menu

class PauseMenu:
    def __init__(self, surface):
        self.surface = surface
        self.font_large = pygame.font.Font(FONT, 70)
        self.font_medium = pygame.font.Font(FONT, 40)
        
        center_x = WIDTH // 2
        center_y = HEIGHT // 2
        button_width = 375
        button_height = 100
        button_spacing = 20
        
        self.resume_button = pygame.Rect(
            center_x - button_width // 2,
            center_y - button_height - button_spacing // 2,
            button_width, button_height
        )
        
        self.back_button = pygame.Rect(
            center_x - button_width // 2,
            center_y + button_spacing // 2,
            button_width, button_height
        )
        
        self.resume_hovered = False
        self.back_hovered = False
    
    def handle_event(self, event, environment):
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            self.resume_hovered = self.resume_button.collidepoint(mouse_pos)
            self.back_hovered = self.back_button.collidepoint(mouse_pos)
        
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            
            if self.resume_button.collidepoint(mouse_pos):
                environment.toggle_pause()
                return True
            
            elif self.back_button.collidepoint(mouse_pos):
                game_state_manager.setState('settings')
                return True
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                environment.toggle_pause()
                return True
        
        return False
    
    def draw(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.surface.blit(overlay, (0, 0))
        
        title_text = self.font_large.render("PAUSED", True, WHITE)
        title_shadow = self.font_large.render("PAUSED", True, BLACK)
        title_x = (WIDTH - title_text.get_width()) // 2
        title_y = HEIGHT // 2 - 250
        self.surface.blit(title_shadow, (title_x + 4, title_y + 4))
        self.surface.blit(title_text, (title_x, title_y))
        
        self._draw_button(self.resume_button, "Resume", self.resume_hovered, (70, 130, 70))
        self._draw_button(self.back_button, "Back", self.back_hovered, (130, 70, 70))
        
        hint_text = self.font_medium.render("Press ESC to Resume", True, (200, 200, 200))
        hint_x = (WIDTH - hint_text.get_width()) // 2
        hint_y = HEIGHT // 2 + 150
        self.surface.blit(hint_text, (hint_x, hint_y))
    
    def _draw_button(self, rect, text, hovered, base_color):
        color = tuple(min(c + 40, 255) for c in base_color) if hovered else base_color
        
        pygame.draw.rect(self.surface, color, rect, border_radius=10)
        pygame.draw.rect(self.surface, WHITE if hovered else (150, 150, 150), rect, 3, border_radius=10)
        
        text_surf = self.font_medium.render(text, True, WHITE)
        shadow_surf = self.font_medium.render(text, True, BLACK)
        
        self.surface.blit(shadow_surf, shadow_surf.get_rect(center=(rect.centerx + 2, rect.centery + 2)))
        self.surface.blit(text_surf, text_surf.get_rect(center=rect.center))