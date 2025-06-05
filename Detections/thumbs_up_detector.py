import mediapipe as mp
from .base_detector import BaseDetector

mp_hands = mp.solutions.hands

class ThumbsUpDetector(BaseDetector):
    def detect(self, hands, frame):
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
        thumb_vertical = thumb_x_spread < 0.08

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
            self.emoji_spawner.spawn_fountain_emojis(frame, "assets/thumbs_up.png")
