# הסוכן - מנהל למידה וקבלת החלטות
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from scripts.ddqn import DDQN
from scripts.replaybuffer import ReplayBuffer, replaybuffer_from_dict

# מימדי הקלט והפלט של הרשת
# מהירות, זוויות ו-30 קרניים (קירות ומכשולים)
STATE_DIM = 33
# (קדימה, אחורה, פניות ושילובים)
ACTION_DIM = 9

class DDQNAgent:    
    def __init__(self, device=None):
        # הגדרת המעבד לחישובים (כרטיס מסך אם זמין, אחרת מעבד רגיל)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # יצירת רשת הלמידה הראשית
        self.policy_net = DDQN(STATE_DIM, ACTION_DIM, device=self.device).to(self.device)
        # יצירת רשת המטרה (עותק)
        self.target_net = DDQN(STATE_DIM, ACTION_DIM, device=self.device).to(self.device)
        # העתקת משקלים מהרשת הראשית לרשת המטרה
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # העברת רשת המטרה למצב חישוב בלבד
        self.target_net.eval()

        # מקדם ניכוי (משקל הגמול העתידי)
        self.gamma = 0.95
        # גודל קבוצת הדגימות בכל שלב למידה
        self.batch_size = 256
        # קצב הלמידה של האופטימייזר
        self.lr = 0.0001
        
        # חקירה
        # 1=100%
        self.epsilon = 1.0
        # .05 = 5%
        self.epsilon_min = 0.05
        # הדעיכה של החקירה ממקסימום למינימום על ידי הכפלה ב-0.9995
        self.epsilon_decay = 0.9995
        
        # כל 200 צעדים לעדכן את רשת המטרה
        self.target_update = 200
        self.episode_count = 0

        # יוצר ריפליי באפר בגודל 250,000 חוויות
        self.replay_buffer = ReplayBuffer(capacity=250000)  
        
        # הגדרת כלי האופטימיזציה אדם המלך
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        # מוריד את קצב הלמידה אם ממוצע התגמולים לא משתפר לאורך זמן
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=500, min_lr=1e-6)

        self.train_step = 0 # מונה צעדי אימון
        self.model_dir = "models" # תיקיית שמירת המודלים
        os.makedirs(self.model_dir, exist_ok=True) # יצירת התיקייה אם אינה קיימת
        self.model_path = os.path.join(self.model_dir, "model.pt") # נתיב הקובץ
        
        self.best_finish_time = 0.0 # שיא מהירות ניצחון
        self.best_finish_episode = 0 # הסבב שבו זה קרה
        self.recent_rewards = [] # ממוצע תגמולים עד 100 סבבים אחרונים

    def get_action(self, state, training=True):
        # בחירת פעולה לפי המצב הנוכחי
        if state is None:
            return 0
        
        # בחירת פעולה אקראית
        if training and random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        else:
            # בחירה לפי הרשת
            with torch.no_grad():
                # המרת המצב לטנזור והוספת מימד Batch
                # [0.5, 0.1, ...] -> tensor([[0.5, 0.1, ...]])
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                # בחירת האינדקס עם הערך הכי גבוה
                # [0.1, 0.5, 0.2] -> .5 -> (אינדקס 1)
                return q_values.max(1)[1].item()

    def update(self):
        # ביצוע צעד למידה אחד מהניסיון שנצבר
        if len(self.replay_buffer) < self.batch_size * 2:
            return None # אין מספיק דגימות ללמידה

        # דגימת חוויות אקראיות מהזיכרון
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # המרה לטנזורים והעברה למעבד הנבחר
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # הגבלת טווח התגמול ליציבות
        rewards = torch.clamp(rewards, -20.0, 20.0)

        # שליפת הערכים שהרשת נתנה לפעולות שבוצעו
        # [[q0, q1...], [q0, q1...]] -> gather -> [q_act1, q_act2]
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # חישוב ערכי המטרה (לוגיקה של למידה כפולה)
        with torch.no_grad():
            # בחירת הפעולה הטובה ביותר מהרשת הראשית
            # [[0.1, 0.8], [0.5, 0.2]] -> [1, 0]
            next_q_values_policy = self.policy_net(next_states)
            best_actions = next_q_values_policy.max(1)[1].unsqueeze(1)
            
            # רשת המטרה מעריכה את שווי הפעולה שנבחרה
            # [[q0, q1...]] -> gather -> [q_best]
            next_q_values_target = self.target_net(next_states)
            next_q_values = next_q_values_target.gather(1, best_actions).squeeze(1)
            
            # חישוב הערך הצפוי לפי משוואת בלמן
            # Target = Reward + (1 - Done) * Gamma * Next_Q
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # חישוב ההפרש (Loss) וביצוע עדכון משקלים
        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.optimizer.zero_grad() # איפוס גרדיאנטים
        loss.backward() # חישוב גרדיאנטים לאחור
        # הגבלת גודל התיקון למניעת קפיצות חדות מדי בלמידה
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step() # עדכון המשקלים

        # עדכון תקופתי של רשת המטרה
        self.train_step += 1
        # שארית חילוק - בודק אם הגענו בדיוק לצעד 200, 400, 600...
        if self.train_step % self.target_update == 0: 
            # העתקת המשקלים
            self.target_net.load_state_dict(self.policy_net.state_dict()) 

        # הפחתת הסיכוי לפעולה אקראית עם התקדמות הלמידה
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item() # החזרת גודל הטעות

    def end_episode(self, episode_reward=0, checkpoints_reached=0, time_remaining=0, finished=False):
        # סיום סבב ועדכון נתונים
        self.episode_count += 1
        
        self.recent_rewards.append(episode_reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0) # שמירת 100 תגמולים אחרונים בלבד
        
        # עדכון שיא אישי במידה והמסלול הושלם מהר יותר
        if finished:
            if time_remaining > self.best_finish_time:
                old_best = self.best_finish_time
                self.best_finish_time = time_remaining
                self.best_finish_episode = self.episode_count
        
        # התאמת קצב הלמידה בצורה אוטומטית לפי ביצועי המודל
        if len(self.recent_rewards) >= 100:
            # אם ממוצע התגמולים לא משתפר, ה-מתזמן יקטין את קצב הלמידה לדיוק טוב יותר
            self.scheduler.step(np.mean(self.recent_rewards))
        
        return self.epsilon

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.model_path
            
        checkpoint = {
            'model_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'episode_count': self.episode_count,
            'best_finish_time': self.best_finish_time,
            'best_finish_episode': self.best_finish_episode,
            'recent_rewards': self.recent_rewards,
            'replay_buffer': self.replay_buffer.to_dict(),
        }
        
        # שמירה בטוחה בעזרת קובץ זמני
        # קרה שמודל שלי נמחק בזמן שמירה
        # אז חיפשתי כיצד לבצע שמירה בטוחה יותר
        tmp = save_path + '.tmp'
        torch.save(checkpoint, tmp)
        
        import time
        for attempt in range(3):
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
                os.rename(tmp, save_path)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1)
                else:
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except:
                            pass

    def load_model(self, filepath=None):
        # טעינת מודל שמור מהקובץ
        if filepath is None:
            filepath = self.model_path
        if not os.path.exists(filepath):
            print("לא נמצא מודל שמור")
            return False

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False) 

        # שחזור מצבי הרשתות והאופטימייזר
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint.get('target_state_dict', checkpoint['model_state_dict']))

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                pass
        
        # שחזור פרמטרי האימון
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.train_step = checkpoint.get('train_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.best_finish_time = checkpoint.get('best_finish_time', 0.0)
        self.best_finish_episode = checkpoint.get('best_finish_episode', 0)
        self.recent_rewards = checkpoint.get('recent_rewards', [])

        if 'replay_buffer' in checkpoint and checkpoint['replay_buffer']:
            self.replay_buffer = replaybuffer_from_dict(checkpoint['replay_buffer'])

        print(f"Loaded model: ep={self.episode_count}, best={self.best_finish_time:.1f}s") # recap על הנעשה 
        return True