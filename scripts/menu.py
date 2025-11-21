import pygame
import sys
from scripts.Constants import *
from scripts.utils import MenuScreen, calculate_ui_constants
from scripts.GameManager import game_state_manager
from pathlib import Path


class Menu:
    def __init__(self, screen, clock):
        pygame.font.init()
        self.screen = screen
        self.clock = clock
        self.UI_CONSTANTS = calculate_ui_constants((WIDTH, HEIGHT))

        # Background
        self.background = pygame.transform.scale(pygame.image.load(MENU), (WIDTH, HEIGHT))

        # Initialize menus
        self.main_menu = MainMenuScreen(self)
        self.settings_menu = RaceSettingsScreen(self)
        self.tester_menu = TesterSettingsScreen(self)  # NEW
        self.active_menu = self.main_menu
        self.main_menu.enable()

    def show_settings_menu(self):
        self.main_menu.disable()
        self.settings_menu.enable()
        self.active_menu = self.settings_menu

    def show_tester_menu(self):  # NEW
        self.main_menu.disable()
        self.tester_menu.enable()
        self.active_menu = self.tester_menu

    def return_to_main(self):
        self.settings_menu.disable()
        self.tester_menu.disable()  # NEW
        self.main_menu.enable()
        self.active_menu = self.main_menu

    def start_game(self):
        game_state_manager.setState('game')

    def start_training(self):
        game_state_manager.training_mode = True
        game_state_manager.setState('training')

    def start_tester(self):  # NEW
        game_state_manager.setState('tester')

    def quit_game(self):
        pygame.time.delay(300)
        pygame.quit()
        sys.exit()

    def run(self):
        self.screen.blit(self.background, (0, 0))

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.quit_game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if self.active_menu in [self.settings_menu, self.tester_menu]:
                    self.return_to_main()

        self.clock.tick(FPS/2)
        self.active_menu.update(events)
        self.active_menu.draw(self.screen)


class MainMenuScreen(MenuScreen):
    def initialize(self):
        self.title = "RACING GAME"
        self.clear_buttons()

        # Center layout
        cx = self.screen.get_width() // 2
        start_y = int(self.screen.get_height() * 0.3)
        width = int(self.screen.get_width() * 0.25)
        spacing = self.UI_CONSTANTS['BUTTON_HEIGHT'] + self.UI_CONSTANTS['BUTTON_SPACING']

        # Create buttons
        buttons = [
            ('PLAY', self.menu.show_settings_menu, None),
            ('TRAIN AI', self.menu.start_training, (70, 100, 180)),
            ('TEST AI', self.menu.show_tester_menu, (180, 100, 70)),  # NEW - Orange color
            ('QUIT', self.menu.quit_game, (200, 50, 50))
        ]

        for i, (text, action, color) in enumerate(buttons):
            self.create_button(text, action, cx - width // 2, start_y + i * spacing, width, color)


class TesterSettingsScreen(MenuScreen):
    """New menu screen for AI tester settings"""
    def __init__(self, menu):
        super().__init__(menu, "AI Performance Test")
        self.info_font = pygame.font.Font(MENUFONT, int(self.screen.get_height() * 0.02))
        self.header_font = pygame.font.Font(MENUFONT, int(self.screen.get_height() * 0.025))
        
        # Tester settings
        self.num_cars = 10
        self.min_cars = 1
        self.max_cars = 100
        
        # Button groups
        self.preset_buttons = []
        self.start_button = None
        self.back_button = None

    def initialize(self):
        self.title = "AI Performance Test"
        self.clear_buttons()
        self.preset_buttons.clear()

        # Layout
        w, h = self.screen.get_size()
        cx = w // 2
        
        btn_width = int(w * 0.12)
        spacing = self.UI_CONSTANTS['BUTTON_HEIGHT'] + int(self.UI_CONSTANTS['BUTTON_SPACING'] * 0.8)
        
        # Preset buttons (horizontal layout)
        presets = [1, 5, 10, 25, 50, 100]
        preset_y = int(h * 0.35)
        total_preset_width = len(presets) * btn_width + (len(presets) - 1) * 20
        start_x = cx - total_preset_width // 2
        
        for i, num in enumerate(presets):
            x = start_x + i * (btn_width + 20)
            btn = self.create_button(
                str(num), 
                lambda n=num: self._set_car_count(n),
                x, 
                preset_y, 
                btn_width,
                (70, 100, 180)
            )
            self.preset_buttons.append(btn)
              
        # Start button
        self.start_button = self.create_button(
            "Test", 
            self._start_test, 
            cx - 150, 
            int(h * 0.85), 
            300, 
            (70, 180, 70)
        )
        
        # Back button
        back_x = int(w * 0.02)
        back_y = int(h * 0.02)
        back_width = int(w * 0.08)
        self.back_button = self.create_button(
            "←", 
            self.menu.return_to_main, 
            back_x, 
            back_y, 
            back_width
        )

    def _set_car_count(self, num):
        """Set car count to specific preset"""
        self.num_cars = max(self.min_cars, min(num, self.max_cars))

    def _start_test(self):
        """Start the tester with current settings"""
        game_state_manager.tester_num_cars = self.num_cars
        self.menu.start_tester()

    def draw(self, surface):
        if not self.enabled:
            return

        # Title
        title = self.title_font.render(self.title, True, COLORS["title"])
        shadow = self.title_font.render(self.title, True, (0, 0, 0))
        cx = (surface.get_width() - title.get_width()) // 2
        ty = int(surface.get_height() * 0.05)
        surface.blit(shadow, (cx + 4, ty + 4))
        surface.blit(title, (cx, ty))

        # Instructions
        w, h = surface.get_size()
        instructions = [
            "Select number of AI cars to test performance",
            "All cars will race simultaneously using trained AI model",
            f"Model will use epsilon = 0.0 (pure exploitation)"
        ]
        
        y = int(h * 0.18)
        for line in instructions:
            text = self.info_font.render(line, True, (200, 200, 200))
            surface.blit(text, text.get_rect(center=(w//2, y)))
            y += 30

        # Section label
        label_y = int(h * 0.28)
        label = self.header_font.render("Preset Amounts", True, (255, 255, 255))
        surface.blit(label, label.get_rect(center=(w//2, label_y)))

        # Current car count display (big and centered)
        count_y = int(h * 0.55)
        count_text = f"{self.num_cars} Cars"
        count_font = pygame.font.Font(MENUFONT, int(h * 0.08))
        count_surf = count_font.render(count_text, True, (255, 215, 0))
        count_shadow = count_font.render(count_text, True, (0, 0, 0))
        
        count_cx = w // 2
        surface.blit(count_shadow, count_shadow.get_rect(center=(count_cx + 3, count_y + 3)))
        surface.blit(count_surf, count_surf.get_rect(center=(count_cx, count_y)))


        # Draw all buttons with highlighting for selected preset
        for i, btn in enumerate(self.preset_buttons):
            preset_value = [1, 5, 10, 25, 50, 100][i]
            selected = self.num_cars == preset_value
            self._draw_button(surface, btn, selected)

        # Draw other buttons normally
        for btn in self.buttons:
            if btn not in self.preset_buttons:
                btn.draw(surface)

    def _draw_button(self, surface, btn, selected):
        """Draw a styled button with selection state"""
        if selected:
            bg = (100, 150, 255)  # Bright blue when selected
        else:
            bg = btn.bg_color if btn.bg_color else (70, 70, 70)
        
        if btn.selected:  # Mouse hover
            bg = tuple(min(c + 30, 255) for c in bg)
        
        pygame.draw.rect(surface, bg, btn.rect, border_radius=btn.border_radius)
        
        border_color = (255, 255, 255) if selected else (200, 200, 200)
        border_width = 3 if selected else 2
        pygame.draw.rect(surface, border_color, btn.rect, border_width, border_radius=btn.border_radius)
        
        text_surf = btn.font.render(btn.text, True, (255, 255, 255))
        surface.blit(text_surf, text_surf.get_rect(center=btn.rect.center))

    def update(self, events):
        """Handle input"""
        if not self.enabled:
            return

        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons:
            btn.update_hover_state(mouse_pos)
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for btn in self.buttons:
                    if btn.selected:
                        btn.action()
                        return


class RaceSettingsScreen(MenuScreen):
    def __init__(self, menu):
        super().__init__(menu, "Race Settings")
        self.car_images = self._load_car_images()
        self.info_font = pygame.font.Font(MENUFONT, int(self.screen.get_height() * 0.02))
        self.header_font = pygame.font.Font(MENUFONT, int(self.screen.get_height() * 0.025))
        
        # Button groups
        self.player1_buttons = []
        self.player2_buttons = []
        self.p1_car_buttons = []
        self.p2_car_buttons = []

    def _load_car_images(self):
        """Load car images once"""
        images = {}
        for color in CAR_COLORS_LIST:
            path = Path(CAR_COLORS[color])
            if path.exists():
                img = pygame.image.load(path)
                img = pygame.transform.rotate(img, 90)
                img = pygame.transform.scale(img, (50, 100))
                images[color] = pygame.transform.scale(img, (100, 50))
        return images

    def initialize(self):
        self.title = "Race Settings"
        self.clear_buttons()
        self.player1_buttons.clear()
        self.player2_buttons.clear()
        self.p1_car_buttons.clear()
        self.p2_car_buttons.clear()

        # Layout
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
        spacing = self.UI_CONSTANTS['BUTTON_HEIGHT'] + int(self.UI_CONSTANTS['BUTTON_SPACING'] * 0.8)

        # Player type buttons
        for i, ptype in enumerate(["Human", "DQN"]):
            y = top + (i + 1) * spacing
            b1 = self.create_button(ptype, lambda pt=ptype: self._toggle_player1(pt), p1_x - btn_width//2, y, btn_width)
            b2 = self.create_button(ptype, lambda pt=ptype: self._toggle_player2(pt), p2_x - btn_width//2, y, btn_width)
            self.player1_buttons.append(b1)
            self.player2_buttons.append(b2)

        # Car color buttons
        for i, color in enumerate(CAR_COLORS_LIST):
            y = top + (i + 1) * spacing
            b1 = self.create_button("", lambda c=color: self._select_p1_car(c), p1_car_x - car_width//2, y, car_width)
            b2 = self.create_button("", lambda c=color: self._select_p2_car(c), p2_car_x - car_width//2, y, car_width)
            self.p1_car_buttons.append((b1, color))
            self.p2_car_buttons.append((b2, color))

        # Start button (center bottom)
        self.create_button("Start", self._start_race, cx - 150, int(h * 0.85), 300, COLORS["start"])
        
        # Back button (top left with arrow)
        back_x = int(w * 0.02)
        back_y = int(h * 0.02)
        back_width = int(w * 0.08)
        self.create_button("←", self.menu.return_to_main, back_x, back_y, back_width)

    # === Player selection ===
    
    def _toggle_player1(self, player_type):
        current = game_state_manager.player1_selection
        game_state_manager.player1_selection = None if current == player_type else player_type

    def _toggle_player2(self, player_type):
        current = game_state_manager.player2_selection
        game_state_manager.player2_selection = None if current == player_type else player_type

    def _select_p1_car(self, color):
        if color != game_state_manager.player2_car_color:
            game_state_manager.player1_car_color = color

    def _select_p2_car(self, color):
        if color != game_state_manager.player1_car_color:
            game_state_manager.player2_car_color = color

    def _start_race(self):
        if game_state_manager.player1_selection or game_state_manager.player2_selection:
            self.menu.start_game()

    # === Drawing - SIMPLIFIED ===
    
    def draw(self, surface):
        if not self.enabled:
            return

        # Title
        title = self.title_font.render(self.title, True, COLORS["title"])
        shadow = self.title_font.render(self.title, True, (0, 0, 0))
        cx = (surface.get_width() - title.get_width()) // 2
        ty = int(surface.get_height() * 0.05)
        surface.blit(shadow, (cx + 4, ty + 4))
        surface.blit(title, (cx, ty))

        # Section labels
        self._draw_labels(surface)
        
        # All buttons
        self._draw_player_buttons(surface)
        self._draw_car_buttons(surface)
        
        # Start/Back buttons
        for btn in self.buttons[-2:]:
            btn.draw(surface)
        
        # Control panels
        self._draw_controls(surface)

    def _draw_labels(self, surface):
        """Draw column headers"""
        w = surface.get_width()
        cx = w // 2
        col_offset = int(w * 0.12)
        car_offset = int(w * 0.16)
        y = int(surface.get_height() * 0.22)

        labels = [
            ("Player 1", cx - col_offset, COLORS["p1"]),
            ("Player 2", cx + col_offset, COLORS["p2"]),
            ("Car", cx - car_offset - col_offset, COLORS["p1"]),
            ("Car", cx + car_offset + col_offset, COLORS["p2"])
        ]

        for text, x, color in labels:
            surf = self.font.render(text, True, color)
            rect = surf.get_rect(center=(x, y))
            surface.blit(surf, rect)

    def _draw_player_buttons(self, surface):
        """Draw player type selection buttons"""
        for i, btn in enumerate(self.player1_buttons):
            ptype = ["Human", "DQN"][i]
            selected = game_state_manager.player1_selection == ptype
            self._draw_button(surface, btn, selected, COLORS["p1"])

        for i, btn in enumerate(self.player2_buttons):
            ptype = ["Human", "DQN"][i]
            selected = game_state_manager.player2_selection == ptype
            self._draw_button(surface, btn, selected, COLORS["p2"])

    def _draw_button(self, surface, btn, selected, color):
        """Draw a styled button"""
        bg = color if selected else COLORS["inactive"]
        pygame.draw.rect(surface, bg, btn.rect)
        pygame.draw.rect(surface, COLORS["border"], btn.rect, 2)
        
        text_color = (255, 255, 255) if selected else (180, 180, 180)
        text = btn.font.render(btn.text, True, text_color)
        surface.blit(text, text.get_rect(center=btn.rect.center))

    def _draw_car_buttons(self, surface):
        """Draw car selection buttons with car images"""
        for btn, color in self.p1_car_buttons:
            selected = game_state_manager.player1_car_color == color
            disabled = game_state_manager.player2_car_color == color
            self._draw_car_btn(surface, btn, color, selected, disabled)

        for btn, color in self.p2_car_buttons:
            selected = game_state_manager.player2_car_color == color
            disabled = game_state_manager.player1_car_color == color
            self._draw_car_btn(surface, btn, color, selected, disabled)

    def _draw_car_btn(self, surface, btn, color, selected, disabled):
        """Draw single car button"""
        bg = COLORS["button_bg"] if selected else COLORS["inactive"]
        pygame.draw.rect(surface, bg, btn.rect)
        
        if disabled:
            overlay = pygame.Surface((btn.rect.width, btn.rect.height))
            overlay.set_alpha(200)
            overlay.fill((51, 51, 51))
            surface.blit(overlay, btn.rect.topleft)
        
        pygame.draw.rect(surface, COLORS["border"], btn.rect, 2)
        
        if color in self.car_images:
            img_rect = self.car_images[color].get_rect(center=btn.rect.center)
            surface.blit(self.car_images[color], img_rect)

    def _draw_controls(self, surface):
        """Draw control panels for active players"""
        w, h = surface.get_size()
        y = int(h * 0.5)
        
        if game_state_manager.player1_selection:
            self._draw_control_panel(surface, int(w * 0.08), y, True)
        
        if game_state_manager.player2_selection:
            self._draw_control_panel(surface, int(w * 0.92), y, False)

    def _draw_control_panel(self, surface, x, y, is_p1):
        """Draw control info panel"""
        controls = {
            True: {'Forward': 'W', 'Backward': 'S', 'Left': 'A', 'Right': 'D'},
            False: {'Forward': 'Up', 'Backward': 'Down', 'Left': 'Left', 'Right': 'Right'}
        }
        
        ctrl = controls[is_p1]
        color = COLORS["p1"] if is_p1 else COLORS["p2"]
        
        # Panel box
        box = pygame.Rect(x - 100, y - 125, 200, 250)
        
        # Layered backdrop
        for i in range(3):
            bg = pygame.Surface((200 - i*2, 250 - i*2))
            bg.set_alpha(100 - i*20)
            bg.fill(color)
            surface.blit(bg, (box.x + i, box.y + i))
        
        pygame.draw.rect(surface, COLORS["border"], box, 3)
        
        # Title
        title = self.info_font.render("Controls", True, COLORS["button_bg"])
        surface.blit(title, title.get_rect(center=(x, box.y + 25)))
        
        # Keys
        small_font = pygame.font.Font(MENUFONT, 12)
        for i, (action, key) in enumerate(ctrl.items()):
            ky = box.y + 80 + i * 45
            
            # Action label
            action_text = small_font.render(action, True, COLORS["button_bg"])
            surface.blit(action_text, action_text.get_rect(center=(x - 45, ky)))
            
            # Key box
            key_rect = pygame.Rect(x + 15, ky - 17, 60, 35)
            key_bg = pygame.Surface((60, 35))
            key_bg.set_alpha(160)
            key_bg.fill(color)
            surface.blit(key_bg, key_rect)
            pygame.draw.rect(surface, COLORS["border"], key_rect, 2)
            
            # Key text
            key_text = small_font.render(key, True, COLORS["button_bg"])
            surface.blit(key_text, key_text.get_rect(center=(x + 45, ky)))

    def update(self, events):
        """Handle input"""
        if not self.enabled:
            return

        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons:
            btn.update_hover_state(mouse_pos)
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for btn in self.buttons:
                    if btn.selected:
                        btn.action()
                        return