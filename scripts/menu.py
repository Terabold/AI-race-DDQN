import pygame
import sys
from scripts.Constants import MENUFONT, WHITE, BLACK, COLORS, CAR_COLORS, CAR_COLORS_LIST, GOLD, WIDTH, HEIGHT
from scripts.utils import Button, calculate_ui_constants
from scripts.GameManager import game_state_manager
from pathlib import Path


class BaseMenuScreen:
    """Base class for all menu screens"""
    
    def __init__(self, screen, title="Menu"):
        self.screen = screen
        self.title = title
        self.UI = calculate_ui_constants((WIDTH, HEIGHT))
        self.font = pygame.font.Font(MENUFONT, 40)
        self.title_font = pygame.font.Font(MENUFONT, 70)
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
            text, action, self.font, self, bg_color
        )
        btn.image = image  # Optional image for car buttons
        btn.disabled = False  # Disabled state
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
        """Universal button drawing - handles all states"""
        # Background color
        if btn.disabled:
            bg = (40, 40, 40)
        elif selected:
            bg = highlight_color or (100, 150, 255)
        else:
            bg = btn.bg_color or (70, 70, 70)
        
        # Hover brightening (only if not disabled)
        if btn.selected and not btn.disabled:
            bg = tuple(min(c + 30, 255) for c in bg)
        
        # Draw background
        pygame.draw.rect(self.screen, bg, btn.rect, border_radius=btn.border_radius)
        
        # Border
        border_color = WHITE if selected and not btn.disabled else (200, 200, 200)
        border_width = 3 if selected else 2
        pygame.draw.rect(self.screen, border_color, btn.rect, border_width, 
                        border_radius=btn.border_radius)
        
        # Content: image or text
        if btn.image:
            img_rect = btn.image.get_rect(center=btn.rect.center)
            self.screen.blit(btn.image, img_rect)
        elif btn.text:
            text_color = (100, 100, 100) if btn.disabled else WHITE
            text_surf = btn.font.render(btn.text, True, text_color)
            self.screen.blit(text_surf, text_surf.get_rect(center=btn.rect.center))
        
        # Disabled overlay with X
        if btn.disabled:
            # X marks
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


class RaceSettingsMenu(BaseMenuScreen):
    
    def __init__(self, screen):
        self.car_images = {}
        self.p1_type_btns = []
        self.p2_type_btns = []
        self.p1_car_btns = []
        self.p2_car_btns = []
        self.info_font = pygame.font.Font(MENUFONT, int(screen.get_height() * 0.02))
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
        
        # Player type buttons
        for i, ptype in enumerate(["Human", "DQN"]):
            y = top + (i + 1) * spacing
            b1 = self.create_button(ptype, lambda pt=ptype: self.toggle_p1(pt), 
                                   p1_x - btn_width//2, y, btn_width)
            b2 = self.create_button(ptype, lambda pt=ptype: self.toggle_p2(pt), 
                                   p2_x - btn_width//2, y, btn_width)
            self.p1_type_btns.append(b1)
            self.p2_type_btns.append(b2)
        
        # Car buttons with images
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
        
        # Start and back
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
        # Update disabled states for car buttons
        for btn in self.p1_car_btns:
            btn.disabled = btn.color_name == game_state_manager.player2_car_color
        for btn in self.p2_car_btns:
            btn.disabled = btn.color_name == game_state_manager.player1_car_color
    
    def draw(self):
        self.draw_title()
        self.draw_labels()
        
        # Player type buttons
        ptypes = ["Human", "DQN"]
        for i, btn in enumerate(self.p1_type_btns):
            selected = game_state_manager.player1_selection == ptypes[i]
            self.draw_button(btn, selected, COLORS["p1"])
        for i, btn in enumerate(self.p2_type_btns):
            selected = game_state_manager.player2_selection == ptypes[i]
            self.draw_button(btn, selected, COLORS["p2"])
        
        # Car buttons
        for btn in self.p1_car_btns:
            selected = btn.color_name == game_state_manager.player1_car_color
            self.draw_button(btn, selected, COLORS["p1"])
        for btn in self.p2_car_btns:
            selected = btn.color_name == game_state_manager.player2_car_color
            self.draw_button(btn, selected, COLORS["p2"])
        
        # Start and back
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
        
        # Background layers
        for i in range(3):
            bg = pygame.Surface((200 - i*2, 250 - i*2))
            bg.set_alpha(100 - i*20)
            bg.fill(color)
            self.screen.blit(bg, (box.x + i, box.y + i))
        
        pygame.draw.rect(self.screen, COLORS["border"], box, 3)
        
        title = self.info_font.render("Controls", True, COLORS["button_bg"])
        self.screen.blit(title, title.get_rect(center=(x, box.y + 25)))
        
        small_font = pygame.font.Font(MENUFONT, 12)
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


class TesterSettingsMenu(BaseMenuScreen):
    
    def __init__(self, screen):
        self.num_cars = 10
        self.preset_btns = []
        self.header_font = pygame.font.Font(MENUFONT, int(screen.get_height() * 0.025))
        self.info_font = pygame.font.Font(MENUFONT, int(screen.get_height() * 0.02))
        self.count_font = pygame.font.Font(MENUFONT, int(screen.get_height() * 0.08))
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
        
        # Instructions
        for i, line in enumerate([
            "Select number of AI cars to test performance",
            "All cars will race simultaneously using trained AI model",
        ]):
            text = self.info_font.render(line, True, (200, 200, 200))
            self.screen.blit(text, text.get_rect(center=(w//2, int(h * 0.18) + i * 30)))
        
        # Section label
        label = self.header_font.render("Preset Amounts", True, WHITE)
        self.screen.blit(label, label.get_rect(center=(w//2, int(h * 0.28))))
        
        # Car count display
        count_y = int(h * 0.55)
        count_text = f"{self.num_cars} Cars"
        count_shadow = self.count_font.render(count_text, True, BLACK)
        count_surf = self.count_font.render(count_text, True, GOLD)
        self.screen.blit(count_shadow, count_shadow.get_rect(center=(w//2 + 3, count_y + 3)))
        self.screen.blit(count_surf, count_surf.get_rect(center=(w//2, count_y)))
        
        # Preset buttons
        for btn in self.preset_btns:
            selected = self.num_cars == btn.preset_value
            self.draw_button(btn, selected)
        
        # Start and back
        for btn in self.buttons[-2:]:
            self.draw_button(btn)