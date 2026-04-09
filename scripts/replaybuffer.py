# חוצץ חוויות - שומר חוויות עבר ללמידה
# דגימה אקראית שוברת קורלציה בין פריימים עוקבים
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=100000):
        # deque מוחק אוטומטית חוויות ישנות כשמתמלא
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # אצווה אקראית ללמידה
        if len(self.buffer) < batch_size:
            indices = range(len(self.buffer))
        else:
            indices = random.sample(range(len(self.buffer)), batch_size)
        
        batch = [self.buffer[i] for i in indices]
        
        # פיצול למערכים
        states = np.vstack([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch], dtype=np.int64)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.vstack([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def to_dict(self):
        # לשמירה לקובץ
        serialized = {'capacity': self.capacity, 'buffer': []}
        for state, action, reward, next_state, done in self.buffer:
            state_ser = state.tolist() if isinstance(state, np.ndarray) else state
            next_state_ser = next_state.tolist() if isinstance(next_state, np.ndarray) else next_state
            serialized['buffer'].append([state_ser, action, reward, next_state_ser, done])
        return serialized


def replaybuffer_from_dict(data):
    # שחזור מ-dict שנשמר
    capacity = data.get('capacity', 10000)
    buffer_data = data.get('buffer', [])
    
    rb = ReplayBuffer(capacity=capacity)
    rb.buffer.clear()
    
    for state_ser, action, reward, next_state_ser, done in buffer_data:
        state_arr = np.array(state_ser, dtype=np.float32)
        next_state_arr = np.array(next_state_ser, dtype=np.float32)
        rb.buffer.append((state_arr, action, reward, next_state_arr, done))
    
    return rb