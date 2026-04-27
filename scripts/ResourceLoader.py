import torch
import pygame
import os
from pathlib import Path

class ResourceManager:
    def __init__(self):
        self.device = None
        
        # מקום בזיכרון לתמונות
        self.images = {}
        
        # מקום בזיכרון לסאונד
        self.sounds = {}
        
        # מקום בזיכרון לפונטים
        self.fonts = {}
        
        # מקום בזיכרון של המסכה של גבולות המסלול
        self.track_border_mask = None
        
        # מודל במצב inference מקום בזיכרון של
        self.model_checkpoint = None
    
    def initialize(self):
        # בודק אם קודה זמין במחשב
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_all(self):
        # טוען את כל המשבים
        try:
            self.load_images()
            self.load_sounds()
            self.load_fonts()
            self.load_model()
        except Exception as e:
            print(f"ResourceManager error: {e}")
    
    def load_images(self):
        # טוען תמונות למשחק
        from scripts.Constants import (
            MENUBG, TRACK, TRACK_BORDER, GRASS, FINISHLINE, BOMB,
            CAR_COLORS, FINISHLINE_SIZE, WIDTH, HEIGHT
        )
        
        try:
            # תפריט
            self.images['menu_bg'] = pygame.transform.scale(
                pygame.image.load(MENUBG), (WIDTH, HEIGHT)
            )
            
            # משחק גרפי
            self.images['track'] = pygame.image.load(TRACK).convert_alpha()
            self.images['grass'] = pygame.image.load(GRASS).convert()
            self.images['bomb'] = pygame.image.load(BOMB).convert_alpha()
            
            # מסכה וגבולות המסלול
            track_border = pygame.image.load(TRACK_BORDER).convert_alpha()
            self.images['track_border'] = track_border
            self.track_border_mask = pygame.mask.from_surface(track_border)
            
            # קו הסיום
            finishline = pygame.image.load(FINISHLINE).convert_alpha()
            self.images['finishline'] = pygame.transform.scale(finishline, FINISHLINE_SIZE)
            self.images['finishline_mask'] = pygame.mask.from_surface(self.images['finishline'])
            
            # טוען תמונות של הרכבים לפי צבעים
            for color, path in CAR_COLORS.items():
                img = pygame.image.load(path).convert_alpha()
                img_rotated = pygame.transform.rotate(img, 90)
                self.images[f'car_{color}'] = pygame.transform.scale(img_rotated, (50, 25))
                self.images[f'car_{color}_original'] = img
        except Exception as e:
            print(f"Image loading error: {e}")
    
    def load_sounds(self):
        # טוען סאונדים של המשחק
        from scripts.Constants import (
            WIN_SOUND, COUNTDOWN_SOUND, COLLIDE_SOUND, 
            OBSTACLE_SOUND, BACKGROUND_MUSIC, DEFAULT_SOUND_VOLUME,
            COLLISION_SOUND_VOLUME, WIN_SOUND_VOLUME, 
            OBSTACLE_SOUND_VOLUME, COUNTDOWN_SOUND_VOLUME
        )
        
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            # סאונדים
            self.sounds['win'] = self.load_sound(WIN_SOUND, WIN_SOUND_VOLUME)
            self.sounds['countdown'] = self.load_sound(COUNTDOWN_SOUND, COUNTDOWN_SOUND_VOLUME)
            self.sounds['collision'] = self.load_sound(COLLIDE_SOUND, COLLISION_SOUND_VOLUME)
            self.sounds['obstacle'] = self.load_sound(OBSTACLE_SOUND, OBSTACLE_SOUND_VOLUME)
            
            # מוזיקה ברקע
            self.sounds['background_music_path'] = BACKGROUND_MUSIC
        except Exception as e:
            print(f"Sound loading error: {e}")
    
    def load_sound(self, path, volume):
        # טוען סאונד בודד עם ווליום
        try:
            sound = pygame.mixer.Sound(str(Path(path)))  # טעינת קובץ סאונד מהנתיב והמרתו לאובייקט סאונד
            sound.set_volume(volume)
            return sound
        except:
            return None
    
    def load_fonts(self):
        # טוען פונטים
        from scripts.Constants import FONT, COUNTDOWN_FONT
        
        try:
            # גדלים של טקסט
            for size in [12, 24, 32, 36, 40, 42, 48, 70, 80, 180]:
                self.fonts[(FONT, size)] = pygame.font.Font(FONT, size)
            
            # ספירת הזינוק
            for size in [180]:
                self.fonts[(COUNTDOWN_FONT, size)] = pygame.font.Font(COUNTDOWN_FONT, size)
        except Exception as e:
            print(f"Font loading error: {e}")
    
    def get_font(self, font_path, size):
        # מביא פונט לפי הנתיב והגודל שלו, אם אין טוען אותו
        key = (font_path, size)
        if key not in self.fonts:
            self.fonts[key] = pygame.font.Font(font_path, size)
        return self.fonts[key]
    
    def load_model(self):
        # טוען מודל למשחק
        try:
            model_path = r"models\UseModel\model_inference.pt"
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model_checkpoint = checkpoint
        except Exception as e:
            self.model_checkpoint = None

# פונקציה עזר לטעינת משאבים חיצוניים בכל מקום בקוד
resource_manager = ResourceManager()