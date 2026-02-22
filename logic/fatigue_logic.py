from collections import deque
from utils.alarm import trigger_alarm

class FatigueDetector:
    def __init__(self):
        self.score_history = deque(maxlen=15)
        self.alarm_on = False

    def evaluate(self, cnn_prob, ear_val, pitch_angle):
        score = 0
        
        # Factor 1: CNN (Deep Learning) - High weight
        if cnn_prob > 0.4: score += 4 
        
        # Factor 2: EAR (Geometric) - Reliable backup
        if ear_val < 0.18: score += 3
        
        # Factor 3: Pitch (Posture) - Drooping head
        if pitch_angle < -15: score += 2

        self.score_history.append(score)
        avg_score = sum(self.score_history) / len(self.score_history)

        # Robust Threshold: Hit a score of 5 on average to trigger
        if avg_score >= 5:
            if not self.alarm_on:
                trigger_alarm()
                self.alarm_on = True
            return "DANGER: FATIGUE", (0, 0, 255)
        
        elif avg_score >= 2:
            self.alarm_on = False
            return "WARNING: DROWSY", (0, 165, 255)
            
        else:
            self.alarm_on = False
            return "ACTIVE", (0, 255, 0)