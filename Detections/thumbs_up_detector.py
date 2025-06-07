import mediapipe as mp
from .base_detector import BaseDetector

mp_hands = mp.solutions.hands

class ThumbsUpDetector(BaseDetector):
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

        # === Thumb joints ===
        thumb_cmc = lm[mp_hands.HandLandmark.THUMB_CMC]
        thumb_mcp = lm[mp_hands.HandLandmark.THUMB_MCP]
        thumb_ip  = lm[mp_hands.HandLandmark.THUMB_IP]
        thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]

        # 1. Thumb is mostly vertical
        thumb_xs = [thumb_cmc.x, thumb_mcp.x, thumb_ip.x, thumb_tip.x]
        thumb_x_spread = max(thumb_xs) - min(thumb_xs)
        thumb_vertical = thumb_x_spread < 0.05

        # 2. Thumb is not curled (tip is clearly above MCP)
        thumb_not_curled = abs(thumb_tip.y - thumb_mcp.y) > 0.1

        # 3. Thumb tip above all other fingertips
        thumb_tip_above = all(
            thumb_tip.y < lm[tip].y
            for tip in [
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP
            ]
        )

        # 4. Other fingers curled
        curled_fingers = all(
            lm[tip].y > lm[mcp].y + 0.01
            for tip, mcp in [
                (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
                (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
                (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
            ]
        )

        if thumb_vertical and thumb_not_curled and thumb_tip_above and curled_fingers:
            if not self.is_effect_active:
                print("ThumbsUpDetector: Thumbs up detected!")
                self.is_effect_active = True
                self.current_effect_frame = self.effect_duration
                self.reaction_manager.set_reaction_active(True) # Set reaction active

    def apply_effect(self, frame, hands_landmarks):
        if self.current_effect_frame > 0:
            # Spawn emojis during the effect duration
            self.emoji_spawner.spawn_fountain_emojis(frame, "assets/thumbs_up.png")
            self.current_effect_frame -= 1
            if self.current_effect_frame == 0:
                self.is_effect_active = False
                self.reaction_manager.set_reaction_active(False) # Set reaction inactive when effect ends
