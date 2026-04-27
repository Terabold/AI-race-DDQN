import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, device=None):
        super(DDQN, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # רשת 3 שכבות פשוטה: 33 -> 128 -> 128 -> 9
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        # המרת הקלט לטנזור והעברה למעבד הנבחר
        if isinstance(x, np.ndarray): # אם הקלט הוא מערך נאמפיי
            x = torch.FloatTensor(x).to(self.device) # המרת המערך לטנזור של פייתורץ והעברה למעבד
        else:
            x = x.to(self.device) # אם הקלט כבר טנזור, העבר אותו למעבד
        
        # מעבר בשכבות ליניאריות עם פונקציית הפעלה
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        
        # שכבת פלט סופית (9 ערכי פעולה)
        return self.fc3(x)
    
    def save(self, filepath):
        # שמירת משקולות הרשת לקובץ
        torch.save(self.state_dict(), filepath)
        
    def load(self, filepath):
        # טעינת משקולות הרשת מהקובץ
        self.load_state_dict(torch.load(filepath, map_location=self.device))