import pygame
import sys
from scripts.Constants import FONT, WHITE, BLACK, COLORS, CAR_COLORS, CAR_COLORS_LIST, WIDTH, HEIGHT
from scripts.utils import Button, calculate_ui_constants
from scripts.GameManager import game_state_manager
from scripts.ResourceLoader import resource_manager

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
        title_center_x = (self.screen.get_width() - title.get_width()) // 2
        title_top_y = int(self.screen.get_height() * 0.05)
        self.screen.blit(shadow, (title_center_x + 4, title_top_y + 4))
        self.screen.blit(title, (title_center_x, title_top_y))
    
    def draw_button(self, btn, selected=False, highlight_color=None):
        if btn.disabled:
            button_color = (40, 40, 40)
        elif selected:
            button_color = highlight_color or (100, 150, 255)
        else:
            button_color = btn.bg_color or (70, 70, 70)
        
        if btn.selected and not btn.disabled:
            # הבהרת צבע הכפתור כשעומדים עליו
            button_color = tuple(min(c + 30, 255) for c in button_color)
        
        pygame.draw.rect(self.screen, button_color, btn.rect, border_radius=btn.border_radius)
        
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
            # X על כפתורים מושבתים
            cross_padding = 8
            cross_start_x, cross_start_y = btn.rect.left + cross_padding, btn.rect.top + cross_padding
            cross_end_x, cross_end_y = btn.rect.right - cross_padding, btn.rect.bottom - cross_padding
            pygame.draw.line(self.screen, (200, 50, 50), (cross_start_x, cross_start_y), (cross_end_x, cross_end_y), 3)
            pygame.draw.line(self.screen, (200, 50, 50), (cross_start_x, cross_end_y), (cross_end_x, cross_start_y), 3)
    
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
        # חישוב מיקומים לפי גודל המסך
        center_x = self.screen.get_width() // 2
        start_y_position = int(self.screen.get_height() * 0.3)
        button_width = int(self.screen.get_width() * 0.25)
        spacing = self.UI['BUTTON_HEIGHT'] + self.UI['BUTTON_SPACING']
        
        buttons = [
            ('PLAY', lambda: game_state_manager.setState('settings'), None),
            ('TRAIN AI', lambda: game_state_manager.setState('training'), (70, 100, 180)),
            ('QUIT', self.quit, (200, 50, 50))
        ]
        
        for i, (text, action, color) in enumerate(buttons):
            self.create_button(text, action, center_x - button_width // 2, start_y_position + i * spacing, button_width, color)
    
    def quit(self):
        pygame.quit()
        sys.exit()


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
            key = f'car_{color}'
            img = resource_manager.images[key]
            self.car_images[color] = pygame.transform.scale(img, (100, 50))
        return self.car_images[color]
    
    def initialize(self):
        self.buttons.clear()
        self.p1_type_btns.clear()
        self.p2_type_btns.clear()
        self.p1_car_btns.clear()
        self.p2_car_btns.clear()
        
        screen_width, screen_height = self.screen.get_size()
        center_x = screen_width // 2
        
        # חישוב רווחים בין עמודות השחקנים
        column_offset_from_center = int(screen_width * 0.12)
        car_selection_offset_from_center = int(screen_width * 0.16)
        
        # מרכזי עמודות לשחקן 1 ו-2
        p1_column_center_x = center_x - column_offset_from_center
        p2_column_center_x = center_x + column_offset_from_center
        
        # מיקומי כפתורי רכב
        p1_car_selection_center_x = center_x - car_selection_offset_from_center - column_offset_from_center
        p2_car_selection_center_x = center_x + car_selection_offset_from_center + column_offset_from_center
        
        button_width = int(screen_width * 0.15)
        car_button_width = int(screen_width * 0.1)
        
        top_margin = int(screen_height * 0.20)
        spacing = self.UI['BUTTON_HEIGHT'] + int(self.UI['BUTTON_SPACING'] * 0.8)
        
        # כפתורי סוג שחקן
        for i, ptype in enumerate(["Human", "DDQN"]):
            y = top_margin + (i + 1) * spacing
            b1 = self.create_button(ptype, lambda pt=ptype: self.toggle_p1(pt), 
                                   p1_column_center_x - button_width//2, y, button_width)
            b2 = self.create_button(ptype, lambda pt=ptype: self.toggle_p2(pt), 
                                   p2_column_center_x - button_width//2, y, button_width)
            self.p1_type_btns.append(b1)
            self.p2_type_btns.append(b2)
        
        # כפתורי רכב
        for i, color in enumerate(CAR_COLORS_LIST):
            y = top_margin + (i + 1) * spacing
            img = self.load_car_image(color)
            b1 = self.create_button("", lambda c=color: self.select_p1_car(c), 
                                   p1_car_selection_center_x - car_button_width//2, y, car_button_width, image=img)
            b2 = self.create_button("", lambda c=color: self.select_p2_car(c), 
                                   p2_car_selection_center_x - car_button_width//2, y, car_button_width, image=img)
            b1.color_name = color
            b2.color_name = color
            self.p1_car_btns.append(b1)
            self.p2_car_btns.append(b2)
        
        self.create_button("Start", self.start, center_x - 150, int(screen_height * 0.85), 300, COLORS["start"])
        self.create_button("←", lambda: game_state_manager.setState('menu'), 
                          int(screen_width * 0.02), int(screen_height * 0.02), int(screen_width * 0.08))
    
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
        # מונע בחירת אותו צבע על ידי שני שחקנים
        for btn in self.p1_car_btns:
            btn.disabled = btn.color_name == game_state_manager.player2_car_color
        for btn in self.p2_car_btns:
            btn.disabled = btn.color_name == game_state_manager.player1_car_color
    
    def draw(self):
        self.draw_title()
        self.draw_labels()
        
        ptypes = ["Human", "DDQN"]
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
        screen_width = self.screen.get_width()
        center_x = screen_width // 2
        column_offset_from_center = int(screen_width * 0.12)
        car_selection_offset_from_center = int(screen_width * 0.16)
        y_position = int(self.screen.get_height() * 0.22)
        
        labels = [
            ("Player1", center_x - column_offset_from_center, COLORS["p1"]),
            ("Player2", center_x + column_offset_from_center, COLORS["p2"]),
            ("Car", center_x - car_selection_offset_from_center - column_offset_from_center, COLORS["p1"]),
            ("Car", center_x + car_selection_offset_from_center + column_offset_from_center, COLORS["p2"])
        ]
        
        for text, x, color in labels:
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, surf.get_rect(center=(x, y_position)))
    
    def draw_controls(self):
        screen_width, screen_height = self.screen.get_size()
        y_position = int(screen_height * 0.5)
        
        if game_state_manager.player1_selection:
            self.draw_control_panel(int(screen_width * 0.08), y_position, True)
        if game_state_manager.player2_selection:
            self.draw_control_panel(int(screen_width * 0.92), y_position, False)
    
    def draw_control_panel(self, x, y, is_p1):
        controls_map = ({'Forward': 'W', 'Backward': 'S', 'Left': 'A', 'Right': 'D'} if is_p1 
                else {'Forward': 'Up', 'Backward': 'Down', 'Left': 'Left', 'Right': 'Right'})
        color = COLORS["p1"] if is_p1 else COLORS["p2"]
        
        box = pygame.Rect(x - 100, y - 125, 200, 250)
        
        for i in range(3):
            # אפקט רקע מוחלש
            panel_background_surface = pygame.Surface((200 - i*2, 250 - i*2))
            panel_background_surface.set_alpha(100 - i*20)
            panel_background_surface.fill(color)
            self.screen.blit(panel_background_surface, (box.x + i, box.y + i))
        
        pygame.draw.rect(self.screen, COLORS["border"], box, 3)
        
        title = self.info_font.render("Controls", True, COLORS["button_bg"])
        self.screen.blit(title, title.get_rect(center=(x, box.y + 25)))
        
        small_font = pygame.font.Font(FONT, 12)
        for i, (action, key) in enumerate(controls_map.items()):
            key_row_y_position = box.y + 80 + i * 45
            
            action_text = small_font.render(action, True, COLORS["button_bg"])
            self.screen.blit(action_text, action_text.get_rect(center=(x - 45, key_row_y_position)))
            
            key_display_rect = pygame.Rect(x + 15, key_row_y_position - 17, 60, 35)
            key_background_surface = pygame.Surface((60, 35))
            key_background_surface.set_alpha(160)
            key_background_surface.fill(color)
            self.screen.blit(key_background_surface, key_display_rect)
            pygame.draw.rect(self.screen, COLORS["border"], key_display_rect, 2)
            
            key_text = small_font.render(key, True, COLORS["button_bg"])
            self.screen.blit(key_text, key_text.get_rect(center=(x + 45, key_row_y_position)))


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
        # יצירת שכבת רקע שקופה
        # pygame, SRCALPHA
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.surface.blit(overlay, (0, 0))
        
        title_text = self.font_large.render("PAUSED", True, WHITE)
        title_shadow = self.font_large.render("PAUSED", True, BLACK)
        title_x = (WIDTH - title_text.get_width()) // 2
        title_y = HEIGHT // 2 - 250
        self.surface.blit(title_shadow, (title_x + 4, title_y + 4))
        self.surface.blit(title_text, (title_x, title_y))
        
        self.draw_button(self.resume_button, "Resume", self.resume_hovered, (70, 130, 70))
        self.draw_button(self.back_button, "Back", self.back_hovered, (130, 70, 70))
        
        hint_text = self.font_medium.render("Press ESC to Resume", True, (200, 200, 200))
        hint_x = (WIDTH - hint_text.get_width()) // 2
        hint_y = HEIGHT // 2 + 150
        self.surface.blit(hint_text, (hint_x, hint_y))
    
    def draw_button(self, rect, text, hovered, base_color):
        # הבהרת הכפתור במעבר עכבר
        color = tuple(min(c + 40, 255) for c in base_color) if hovered else base_color
        
        pygame.draw.rect(self.surface, color, rect, border_radius=10)
        pygame.draw.rect(self.surface, WHITE if hovered else (150, 150, 150), rect, 3, border_radius=10)
        
        text_surf = self.font_medium.render(text, True, WHITE)
        shadow_surf = self.font_medium.render(text, True, BLACK)
        
        self.surface.blit(shadow_surf, shadow_surf.get_rect(center=(rect.centerx + 2, rect.centery + 2)))
        self.surface.blit(text_surf, text_surf.get_rect(center=rect.center))