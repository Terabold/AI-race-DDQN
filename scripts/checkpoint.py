import pygame
from scripts.Constants import TRACK_CHECKPOINT_ZONES, FONT

# פונקציה הבודקת חיתוך בין שני קטעים (מסלול הרכב וקו הצ'קפוינט)
def lines_intersect(p1, p2, p3, p4):
    x1, y1 = p1 # מיקום הרכב הקודם
    x2, y2 = p2 # מיקום הרכב הנוכחי
    x3, y3 = p3 # נקודה 1 של הצ'קפוינט
    x4, y4 = p4 # נקודה 2 של הצ'קפוינט
    
    # חישוב המכנה לפי נוסחת דטרמיננטה (למציאת חיתוך בין קווים)
    den = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(den) < 1e-10: # אם המכנה אפס, הקווים מקבילים
        return False
    
    # חישוב מיקום נקודת החיתוך היחסית על פני שני הקווים
    ua = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / den
    ub = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / den
    
    # אם שני המקדמים בין 0 ל-1, נקודת החיתוך נמצאת בתוך שני הקטעים
    return 0 <= ua <= 1 and 0 <= ub <= 1

class CheckpointManager:
    def __init__(self):
        self.zones = TRACK_CHECKPOINT_ZONES
        self.total_checkpoints = len(self.zones)
        self.font = pygame.font.Font(FONT, 12)
        self.reset()

    def reset(self):
        self.current_idx = 0
        self.crossed_count = 0
        self.cross_counts = [0] * self.total_checkpoints
        self.prev_pos = None

    def check_crossing(self, car_pos):
        if self.prev_pos is None:
            self.prev_pos = car_pos
            return False, False

        # בדיקת חצייה קדימה (של הצ'קפוינט הבא בתור)
        if self.current_idx < self.total_checkpoints:
            p1, p2 = self.zones[self.current_idx]
            if lines_intersect(self.prev_pos, car_pos, p1, p2):
                self.cross_counts[self.current_idx] += 1 # סימון שהקו הזה נחצה
                self.crossed_count += 1 # עדכון סך כל החציות במסלול
                self.current_idx += 1 # קידום היעד לצ'קפוינט הבא
                self.prev_pos = car_pos
                return True, False # מחזיר: אמת (התקדמנו), שקר (לא חזרנו אחורה)

        # בדיקת נסיעה לאחור (חצייה מחדש של צ'קפוינטים קודמים)
        for i in range(self.current_idx):
            p1, p2 = self.zones[i]
            if lines_intersect(self.prev_pos, car_pos, p1, p2):
                self.cross_counts[i] += 1 # סימון חצייה נוספת של קו ישן
                self.prev_pos = car_pos
                return False, True # מחזיר: שקר (לא התקדמנו), אמת (חזרנו אחורה)

        self.prev_pos = car_pos
        return False, False

    def draw(self, surface):
        for i, (p1, p2) in enumerate(self.zones):
            if i == self.current_idx:
                color, width = (0, 255, 0), 4 # הבא בתור - ירוק
            elif i < self.current_idx:
                # עברנו - אפור, או אדום אם חזרנו אחורה וסימנו שוב
                color = (255, 0, 0) if self.cross_counts[i] > 1 else (100, 100, 100)
                width = 3 if self.cross_counts[i] > 1 else 2
            else:
                color, width = (0, 100, 0), 2 # טרם הושג - ירוק כהה
            
            pygame.draw.line(surface, color, p1, p2, width)
            # מציאת נקודת המרכז של הקו כדי להציג בה את הטקסט
            cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            
            if self.cross_counts[i] > 0:
                txt = self.font.render(f"x{self.cross_counts[i]}", True, (255, 255, 255))
                surface.blit(txt, (cx + 10, cy - 10))
