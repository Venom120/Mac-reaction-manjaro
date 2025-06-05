import mediapipe as mp
from .base_detector import BaseDetector

mp_hands = mp.solutions.hands

class PeaceDetector(BaseDetector):
    def detect(self, hands, frame):
        if not hands or len(hands) != 1:
            return

        lm = hands[0].landmark

        # 1. Index and middle finger extended (tip above MCP and PIP)
        index_extended = (
            lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        )

        middle_extended = (
            lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
        )

        # 2. Ring and pinky curled (tip clearly below MCP)
        ring_curled = lm[mp_hands.HandLandmark.RING_FINGER_TIP].y > lm[mp_hands.HandLandmark.RING_FINGER_MCP].y + 0.02
        pinky_curled = lm[mp_hands.HandLandmark.PINKY_TIP].y > lm[mp_hands.HandLandmark.PINKY_MCP].y + 0.02

        # 3. Optional: Thumb not interfering (thumb tip not above index tip)
        thumb_neutral = lm[mp_hands.HandLandmark.THUMB_TIP].y > lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

        if index_extended and middle_extended and ring_curled and pinky_curled and thumb_neutral:
            self.emoji_spawner.spawn_fountain_emojis(frame, "assets/peace.png")
