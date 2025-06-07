import mediapipe as mp
import cv2
import math
from Detections.base_detector import BaseDetector
from Detections.face_detector import FaceDetector

mp_hands = mp.solutions.hands

class BlushDetector(BaseDetector):
    def __init__(self, emoji_spawner, reaction_manager):
        super().__init__(emoji_spawner)
        self.reaction_manager = reaction_manager # Store the ReactionManager instance
        self.blush_active = False
        self.blush_duration = 40  # frames
        self.blush_timer = 0
        self.face_detector = FaceDetector()

    def detect(self, hands, frame):
        # If another reaction is active, do not perform detection
        if self.reaction_manager.is_reaction_active():
            return

        faces = self.face_detector.detect(frame)

        if not faces or len(hands) != 2:
            self.blush_active = False
            return

        tip_x_1 = hands[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        tip_x_2 = hands[1].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x

        if tip_x_1 < tip_x_2:
            left_hand = hands[0].landmark
            right_hand = hands[1].landmark
        else:
            left_hand = hands[1].landmark
            right_hand = hands[0].landmark

        h, w, _ = frame.shape
        def px(lm): return int(lm.x * w), int(lm.y * h)

        tip1, mcp1 = left_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP], left_hand[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        tip2, mcp2 = right_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP], right_hand[mp_hands.HandLandmark.INDEX_FINGER_MCP]

        x1, y1 = px(tip1)
        x2, y2 = px(tip2)
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance > 50:
            self.blush_active = False
            return

        dir1_x = tip1.x - mcp1.x
        dir2_x = tip2.x - mcp2.x
        pointing_inward = dir1_x > 0 and dir2_x < 0

        def fingers_curled(lm, hand_type):
            if hand_type == 'left':
                return all(
                    lm[tip].x + 0.01 < lm[mcp].x
                    for tip, mcp in [
                        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
                        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
                    ]
                )
            else:
                return all(
                    lm[tip].x > lm[mcp].x - 0.01
                    for tip, mcp in [
                        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
                        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
                    ]
                )

        if pointing_inward and fingers_curled(left_hand, 'left') and fingers_curled(right_hand, 'right'):
            if not self.blush_active:
                print("BlushDetector: Blush gesture detected!")
                self.blush_active = True
                self.blush_timer = self.blush_duration
                self.reaction_manager.set_reaction_active(True) # Set reaction active
        else:
            self.blush_active = False


    def apply_effect(self, frame, hands_landmarks):
        if self.blush_active and self.blush_timer > 0:
            # Cancel blush if hands are gone even though face is present
            if not hands_landmarks or len(hands_landmarks) < 2:
                self.blush_timer = 0
                self.blush_active = False
                self.reaction_manager.set_reaction_active(False) # Set reaction inactive
                return

            faces = self.face_detector.detect(frame)
            if faces:
                self.draw_blush(frame, faces)
            self.blush_timer -= 1
            if self.blush_timer <= 0:
                self.blush_active = False
                self.reaction_manager.set_reaction_active(False) # Set reaction inactive
        elif self.blush_active and self.blush_timer <= 0:
             # This case handles when blush_active is True but timer ran out in a previous frame
             # Ensure reaction is set to inactive
             self.blush_active = False
             self.reaction_manager.set_reaction_active(False)

    def draw_blush(self, frame, faces):
        if not faces:
            return

        face = faces[0]
        keypoints = face["keypoints"]
        if len(keypoints) < 4:
            return

        right_eye = keypoints[0]
        left_eye = keypoints[1]
        mouth_center = keypoints[3]

        right_cheek_x = int((right_eye[0] + mouth_center[0]) / 2 + (right_eye[0] - mouth_center[0]) * 0.5) - 20
        right_cheek_y = int((right_eye[1] + mouth_center[1]) / 2)
        left_cheek_x = int((left_eye[0] + mouth_center[0]) / 2 + (left_eye[0] - mouth_center[0]) * 0.5) + 20
        left_cheek_y = int((left_eye[1] + mouth_center[1]) / 2)

        overlay = frame.copy()
        blush_color = (147, 20, 255)
        radius = 25
        alpha = 0.5
        blur = (49, 49)

        cv2.circle(overlay, (right_cheek_x, right_cheek_y), radius, blush_color, -1)
        cv2.circle(overlay, (left_cheek_x, left_cheek_y), radius, blush_color, -1)

        blurred = cv2.GaussianBlur(overlay, blur, 0)
        cv2.addWeighted(blurred, alpha, frame, 1 - alpha, 0, frame)
