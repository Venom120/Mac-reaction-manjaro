import math
import random
from reactions import Emoji
import mediapipe as mp
mp_hands = mp.solutions.hands

def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def angle_between(p1, center, p2):
    a = (p1.x - center.x, p1.y - center.y)
    b = (p2.x - center.x, p2.y - center.y)
    dot = a[0]*b[0] + a[1]*b[1]
    det = a[0]*b[1] - a[1]*b[0]
    angle = math.atan2(det, dot)
    return math.degrees(abs(angle))

class HeartDetector:
    def __init__(self, emoji_spawner, reaction_manager):
        self.emoji_spawner = emoji_spawner
        self.reaction_manager = reaction_manager # Store the ReactionManager instance
        self.is_effect_active = False
        self.effect_duration = 30 # frames (adjust as needed)
        self.current_effect_frame = 0

    def detect(self, hands, frame):
        # If another reaction is active, do not perform detection
        if self.reaction_manager.is_reaction_active():
            return

        if len(hands) != 2:
            return

        h, w = frame.shape[:2]
        hand1 = hands[0].landmark
        hand2 = hands[1].landmark

        thumb1 = hand1[mp_hands.HandLandmark.THUMB_TIP]
        index1_tip = hand1[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index1_pip = hand1[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        wrist1 = hand1[mp_hands.HandLandmark.WRIST]

        thumb2 = hand2[mp_hands.HandLandmark.THUMB_TIP]
        index2_tip = hand2[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index2_pip = hand2[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        wrist2 = hand2[mp_hands.HandLandmark.WRIST]

        # Wrist distance check (normalized units, adjust threshold as needed)
        if distance(wrist1, wrist2) < 0.25:
            return

        # 1. Check fingertip distances
        if distance(thumb1, thumb2) > 0.09 or distance(index1_tip, index2_tip) > 0.09:
            return

        # 2. Check dip of index fingers (both PIP below TIP)
        if not (index1_pip.y < index1_tip.y and index2_pip.y < index2_tip.y):
            return

        # 3. Angle V-shape check (optional but helps accuracy)
        angle1 = angle_between(thumb1, index1_tip, wrist1)
        angle2 = angle_between(thumb2, index2_tip, wrist2)
        if not (30 < angle1 < 65 and 30 < angle2 < 65):
            return

        # If all checks pass and no effect is active, trigger the effect
        if not self.is_effect_active:
            print("HeartDetector: Heart gesture detected!")
            self.is_effect_active = True
            self.current_effect_frame = self.effect_duration
            self.reaction_manager.set_reaction_active(True) # Set reaction active

    def apply_effect(self, frame, hands_landmarks):
        if self.current_effect_frame > 0:
            # 4. Center between thumb tips for spray origin
            # Recalculate center based on current hand positions if available
            if hands_landmarks and len(hands_landmarks) == 2:
                h, w = frame.shape[:2]
                hand1 = hands_landmarks[0].landmark
                hand2 = hands_landmarks[1].landmark
                thumb1 = hand1[mp_hands.HandLandmark.THUMB_TIP]
                thumb2 = hand2[mp_hands.HandLandmark.THUMB_TIP]
                cx = int((thumb1.x + thumb2.x) / 2 * w - 20)
                cy = int((thumb1.y + thumb2.y) / 2 * h - 50)

                # Spawn emojis during the effect duration
                angle = random.uniform(-0.8, 0.8)
                speed = random.uniform(5, 7)
                vx = speed * math.sin(angle)
                vy = -speed * math.cos(angle)
                scale = random.uniform(0.4, 0.8)
                self.emoji_spawner.active_emojis.append(
                    Emoji(cx, cy, vx, vy, scale, "assets/heart.png")
                )

            self.current_effect_frame -= 1
            if self.current_effect_frame == 0:
                self.is_effect_active = False
                self.reaction_manager.set_reaction_active(False) # Set reaction inactive when effect ends
