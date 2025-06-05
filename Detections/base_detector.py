import mediapipe as mp

mp_hands = mp.solutions.hands

class BaseDetector:
    def __init__(self, emoji_spawner):
        self.emoji_spawner = emoji_spawner

    def detect(self, hands, frame):
        pass