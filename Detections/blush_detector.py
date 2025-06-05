import mediapipe as mp
import cv2
import math
from Detections.base_detector import BaseDetector

mp_hands = mp.solutions.hands

class BlushDetector(BaseDetector):
    def __init__(self, emoji_spawner):
        super().__init__(emoji_spawner)
        self.blush_active = False
        self.blush_duration = 10  # frames
        self.blush_timer = 0

    def detect(self, hands, frame):
        if len(hands) != 2:
            self.blush_active = False
            return

        # Determine which hand is on the left (lower x-value)
        # Use index finger tip for comparison
        tip_x_hand1 = hands[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        tip_x_hand2 = hands[1].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x

        if tip_x_hand1 < tip_x_hand2:
            hand1_landmarks = hands[0].landmark
            hand2_landmarks = hands[1].landmark
        else:
            hand1_landmarks = hands[1].landmark
            hand2_landmarks = hands[0].landmark

        h, w, _ = frame.shape

        def to_pixel_coords(lm):
            return int(lm.x * w), int(lm.y * h)

        tip1 = hand1_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        mcp1 = hand1_landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        tip2 = hand2_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        mcp2 = hand2_landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

        x1, y1 = to_pixel_coords(tip1)
        x2, y2 = to_pixel_coords(tip2)

        # 1. Are index fingers close together?
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance > 50:
            self.blush_active = False
            return

        # 2. Are both index fingers pointing inward?
        def direction_vector(p1, p2):
            return p2.x - p1.x, p2.y - p1.y

        dir1_x, _ = direction_vector(mcp1, tip1)
        dir2_x, _ = direction_vector(mcp2, tip2)

        # Check direction based on the assigned hand1 and hand2
        pointing_inward = dir1_x > 0 and dir2_x < 0

        # 3. All other fingers curled (based on your logic)
        def fingers_curled(lm, hand_type):
            # hand_type is 'left' or 'right' based on which hand is hand1 or hand2
            if hand_type == 'left': # This corresponds to hand1 (lower x)
                 result = all(
                    lm[tip].x + 0.01 < lm[mcp].x
                    for tip, mcp in [
                        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
                        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
                    ]
                )
                 return result
            elif hand_type == 'right': # This corresponds to hand2 (higher x)
                result = all(
                    lm[tip].x > lm[mcp].x - 0.01 # Corrected logic for inward curl on the right hand
                    for tip, mcp in [
                        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
                        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
                    ]
                )
                return result
            return False # Should not happen with other than left or right

        curled_fingers = fingers_curled(hand1_landmarks, 'left') and fingers_curled(hand2_landmarks, 'right')
        if pointing_inward and curled_fingers:
            self.blush_active = True
            self.blush_timer = self.blush_duration
        else:
            self.blush_active = False

        # Draw blush effect
        if self.blush_active and self.blush_timer > 0:
            self.draw_blush(frame)
            self.blush_timer -= 1
        else:
            self.blush_active = False

    def draw_blush(self, frame):
        h, w, _ = frame.shape
        # Left cheek
        cv2.circle(frame, (int(w * 0.35), int(h * 0.55)), 25, (147, 20, 255), -1)
        # Right cheek
        cv2.circle(frame, (int(w * 0.65), int(h * 0.55)), 25, (147, 20, 255), -1)
