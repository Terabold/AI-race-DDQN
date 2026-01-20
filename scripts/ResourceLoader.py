# Universal ResourceManager - loads ALL assets once at startup
import torch
import pygame
import os
from pathlib import Path

class ResourceManager:
    """Central manager for ALL game assets - images, sounds, fonts, masks, models"""
    
    def __init__(self):
        self.device = None
        
        # Images
        self.images = {}
        
        # Sounds  
        self.sounds = {}
        
        # Fonts (cached by size)
        self.fonts = {}
        
        # Track collision mask (precomputed)
        self.track_border_mask = None
        
        # Model
        self.model_checkpoint = None
    
    def initialize(self):
        """Initialize device"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_all(self):
        """Load ALL assets - images, sounds, fonts, model"""
        try:
            self._load_images()
            self._load_sounds()
            self._load_fonts()
            self._load_model()
        except Exception as e:
            print(f"ResourceManager error: {e}")
    
    def _load_images(self):
        """Load all game images"""
        from scripts.Constants import (
            MENUBG, TRACK, TRACK_BORDER, GRASS, FINISHLINE, BOMB,
            CAR_COLORS, FINISHLINE_SIZE, WIDTH, HEIGHT
        )
        
        try:
            # Menu
            self.images['menu_bg'] = pygame.transform.scale(
                pygame.image.load(MENUBG), (WIDTH, HEIGHT)
            )
            
            # Game assets
            self.images['track'] = pygame.image.load(TRACK).convert_alpha()
            self.images['grass'] = pygame.image.load(GRASS).convert()
            self.images['bomb'] = pygame.image.load(BOMB).convert_alpha()
            
            # Track border + precompute collision mask
            track_border = pygame.image.load(TRACK_BORDER).convert_alpha()
            self.images['track_border'] = track_border
            self.track_border_mask = pygame.mask.from_surface(track_border)
            
            # Finish line (pre-scaled)
            finishline = pygame.image.load(FINISHLINE).convert_alpha()
            self.images['finishline'] = pygame.transform.scale(finishline, FINISHLINE_SIZE)
            self.images['finishline_mask'] = pygame.mask.from_surface(self.images['finishline'])
            
            # Car images (pre-rotated and scaled for menu display)
            for color, path in CAR_COLORS.items():
                img = pygame.image.load(path).convert_alpha()
                img_rotated = pygame.transform.rotate(img, 90)
                self.images[f'car_{color}'] = pygame.transform.scale(img_rotated, (50, 25))
                # Also store original for gameplay (will be scaled differently by Car class)
                self.images[f'car_{color}_original'] = img
        except Exception as e:
            print(f"Image loading error: {e}")
    
    def _load_sounds(self):
        """Load all game sounds"""
        from scripts.Constants import (
            WIN_SOUND, COUNTDOWN_SOUND, COLLIDE_SOUND, 
            OBSTACLE_SOUND, BACKGROUND_MUSIC, DEFAULT_SOUND_VOLUME,
            COLLISION_SOUND_VOLUME, WIN_SOUND_VOLUME, 
            OBSTACLE_SOUND_VOLUME, COUNTDOWN_SOUND_VOLUME
        )
        
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            # Sound effects
            self.sounds['win'] = self._load_sound(WIN_SOUND, WIN_SOUND_VOLUME)
            self.sounds['countdown'] = self._load_sound(COUNTDOWN_SOUND, COUNTDOWN_SOUND_VOLUME)
            self.sounds['collision'] = self._load_sound(COLLIDE_SOUND, COLLISION_SOUND_VOLUME)
            self.sounds['obstacle'] = self._load_sound(OBSTACLE_SOUND, OBSTACLE_SOUND_VOLUME)
            
            # Background music path (pygame.mixer.music uses path, not Sound object)
            self.sounds['background_music_path'] = BACKGROUND_MUSIC
        except Exception as e:
            print(f"Sound loading error: {e}")
    
    def _load_sound(self, path, volume):
        """Load a single sound with volume"""
        try:
            sound = pygame.mixer.Sound(str(Path(path)))
            sound.set_volume(volume)
            return sound
        except:
            return None
    
    def _load_fonts(self):
        """Load common fonts"""
        from scripts.Constants import FONT, COUNTDOWN_FONT
        
        try:
            # Cache commonly used font sizes
            for size in [12, 24, 32, 36, 40, 42, 48, 70, 80, 180]:
                self.fonts[(FONT, size)] = pygame.font.Font(FONT, size)
            
            # Countdown font
            for size in [180]:
                self.fonts[(COUNTDOWN_FONT, size)] = pygame.font.Font(COUNTDOWN_FONT, size)
        except Exception as e:
            print(f"Font loading error: {e}")
    
    def get_font(self, font_path, size):
        """Get font from cache, load if not cached"""
        key = (font_path, size)
        if key not in self.fonts:
            self.fonts[key] = pygame.font.Font(font_path, size)
        return self.fonts[key]
    
    def _load_model(self):
        """Load inference model for AI gameplay"""
        try:
            model_path = r"models\UseModel\model_inference.pt"
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model_checkpoint = checkpoint
        except Exception as e:
            self.model_checkpoint = None

# Global instance
resource_manager = ResourceManager()
