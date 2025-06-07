import mediapipe as mp
from .base_detector import BaseDetector

mp_hands = mp.solutions.hands

class PeaceDetector(BaseDetector):
    def __init__(self, emoji_spawner, reaction_manager):
        super().__init__(emoji_spawner)
        self.reaction_manager = reaction_manager # Store the ReactionManager instance
        self.is_effect_active = False
        self.effect_duration = 15 # frames (adjust as needed)
        self.current_effect_frame = 0

    def detect(self, hands, frame):
        # If another reaction is active, do not perform detection
        if self.reaction_manager.is_reaction_active():
            return

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
            if not self.is_effect_active:
                print("PeaceDetector: Peace sign detected!")
                self.is_effect_active = True
                self.current_effect_frame = self.effect_duration
                self.reaction_manager.set_reaction_active(True) # Set reaction active

    def apply_effect(self, frame, hands_landmarks):
        if self.current_effect_frame > 0:
            # Spawn emojis during the effect duration
            self.emoji_spawner.spawn_fountain_emojis(frame, "assets/peace.png")
            self.current_effect_frame -= 1
            if self.current_effect_frame == 0:
                self.is_effect_active = False
                self.reaction_manager.set_reaction_active(False) # Set reaction inactive when effect ends
