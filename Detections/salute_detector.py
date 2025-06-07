import mediapipe as mp
from .base_detector import BaseDetector
from .face_detector import FaceDetector # Import FaceDetector
import cv2
import numpy as np

mp_hands = mp.solutions.hands

class SaluteDetector(BaseDetector):
    def __init__(self, emoji_spawner, reaction_manager):
        super().__init__(emoji_spawner)
        self.reaction_manager = reaction_manager
        self.is_effect_active = False
        self.effect_duration = 60 # frames
        self.current_effect_frame = 0
        self.salute_image_path = "assets/salute.png"
        self.salute_image = cv2.imread(self.salute_image_path, cv2.IMREAD_UNCHANGED)
        if self.salute_image is None:
            print(f"Error loading image: {self.salute_image_path}")
        else:
            # Ensure image has an alpha channel for transparency
            if self.salute_image.shape[2] == 3:
                self.salute_image = cv2.cvtColor(self.salute_image, cv2.COLOR_BGR2BGRA)

        self.face_detector = FaceDetector() # Instantiate FaceDetector
        super().__init__(emoji_spawner)
        self.reaction_manager = reaction_manager
        self.is_effect_active = False
        self.effect_duration = 60 # frames
        self.current_effect_frame = 0
        # Thresholds for salute detection
        self.hand_flat_horizontal_threshold = 0.02
        self.wrist_near_eye_level_threshold = 0.04
        self.wrist_aligned_with_face_threshold = 0.45
        self.salute_image_path = "assets/salute.png"
        self.salute_image = cv2.imread(self.salute_image_path, cv2.IMREAD_UNCHANGED)
        if self.salute_image is None:
            print(f"Error loading image: {self.salute_image_path}")
        else:
            # Ensure image has an alpha channel for transparency
            if self.salute_image.shape[2] == 3:
                self.salute_image = cv2.cvtColor(self.salute_image, cv2.COLOR_BGR2BGRA)


    def detect(self, hands, frame): # Removed 'faces' parameter
        is_salute = False # Flag to indicate if salute is currently detected

        if self.reaction_manager.is_reaction_active() and not self.is_effect_active:
            # If another reaction is active and this effect is not, do not perform detection
            return

        # Perform face detection using the internal FaceDetector
        faces = self.face_detector.detect(frame)

        # Check if one hand and at least one face are detected
        if hands and len(hands) == 1 and faces and len(faces) > 0: # Check if faces list is not empty
            hand_lm = hands[0].landmark
            face_data = faces[0] # Get the first detected face data (a dictionary)

            # Get hand landmarks
            index_tip = hand_lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = hand_lm[mp_hands.HandLandmark.WRIST]
            index_mcp = hand_lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            # Get face keypoints (eyes) from the face data dictionary
            # Indices 0 and 1 are typically right and left eye respectively for MediaPipe Face Detection keypoints
            face_keypoints = face_data["keypoints"]
            right_eye_kp = face_keypoints[0]
            left_eye_kp = face_keypoints[1]

            # Convert keypoint tuples to normalized coordinates for consistency with hand landmarks
            frame_height, frame_width, _ = frame.shape
            left_eye_norm = type('obj', (object,), {'x': left_eye_kp[0] / frame_width, 'y': left_eye_kp[1] / frame_height})()
            right_eye_norm = type('obj', (object,), {'x': right_eye_kp[0] / frame_width, 'y': right_eye_kp[1] / frame_height})()


            # Calculate average eye level and face center x-coordinate using normalized coordinates
            avg_eye_y = (left_eye_norm.y + right_eye_norm.y) / 2
            face_center_x = (left_eye_norm.x + right_eye_norm.x) / 2

            # Refined checks for a potential salute pose
            # 1. Hand is relatively flat (e.g., index finger tip and wrist are somewhat aligned horizontally)
            hand_flat_horizontal = abs(index_tip.y - wrist.y) < self.hand_flat_horizontal_threshold

            # 2. Wrist is near the average eye level
            wrist_near_eye_level = abs(wrist.y - avg_eye_y) < self.wrist_near_eye_level_threshold # Threshold may need tuning

            # 3. Wrist is horizontally aligned with the face center
            wrist_aligned_with_face = abs(wrist.x - face_center_x) < self.wrist_aligned_with_face_threshold # Threshold may need tuning

            # 4. Fingers are relatively straight (e.g., index finger tip is above its MCP)
            fingers_straight = index_tip.y < index_mcp.y

            # Combine conditions
            if hand_flat_horizontal and wrist_near_eye_level and fingers_straight and wrist_aligned_with_face:
                 is_salute = True
        if is_salute and not self.is_effect_active:
            print("SaluteDetector: Salute detected!")
            self.is_effect_active = True
            self.current_effect_frame = self.effect_duration
            self.reaction_manager.set_reaction_active(True)
        elif not is_salute and self.is_effect_active:
             # If salute is no longer detected and effect is active, start fading out
             # The fade out will be handled in apply_effect by the frame counter
             pass # No change needed here, apply_effect handles the countdown
             pass # No change needed here, apply_effect handles the countdown


    def apply_effect(self, frame, hands_landmarks, faces_landmarks): # Keep faces_landmarks parameter for consistency with reactions.py call
        if self.is_effect_active and self.salute_image is not None:
            frame_height, frame_width, _ = frame.shape
            img_height, img_width, _ = self.salute_image.shape

            # Resize image to cover the whole frame while maintaining aspect ratio
            scale_width = frame_width / img_width
            scale_height = frame_height / img_height
            scale = max(scale_width, scale_height)

            target_width = int(img_width * scale)
            target_height = int(img_height * scale)

            resized_salute_image = cv2.resize(self.salute_image, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # Calculate the position to center the resized image
            x_offset = (frame_width - target_width) // 2
            y_offset = (frame_height - target_height) // 2

            # Determine if salute gesture is still present
            is_salute_present = False
            # Check if one hand and at least one face are detected (using the passed faces_landmarks)
            if hands_landmarks and len(hands_landmarks) == 1 and faces_landmarks and len(faces_landmarks) > 0: # Check if faces_landmarks list is not empty
                hand_lm = hands_landmarks[0].landmark
                face_detection = faces_landmarks[0] # Get the first detected face (a MediaPipe Detection object)

                # Get hand landmarks
                index_tip = hand_lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = hand_lm[mp_hands.HandLandmark.WRIST]
                index_mcp = hand_lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                # Get face landmarks (eyes) from the MediaPipe Detection object
                # Indices 0 and 1 are typically right and left eye respectively for MediaPipe Face Detection keypoints
                right_eye = face_detection.location_data.relative_keypoints[0]
                left_eye = face_detection.location_data.relative_keypoints[1]

                # Calculate average eye level and face center x-coordinate using normalized coordinates
                avg_eye_y = (left_eye.y + right_eye.y) / 2
                face_center_x = (left_eye.x + right_eye.x) / 2

                # Refined checks for a potential salute pose (same as in detect)
                hand_flat_horizontal = abs(index_tip.y - wrist.y) < self.hand_flat_horizontal_threshold
                wrist_near_eye_level = abs(wrist.y - avg_eye_y) < self.wrist_near_eye_level_threshold # Threshold may need tuning
                wrist_aligned_with_face = abs(wrist.x - face_center_x) < self.wrist_aligned_with_face_threshold # Threshold may need tuning

                fingers_straight = index_tip.y < index_mcp.y

                if hand_flat_horizontal and wrist_near_eye_level and fingers_straight and wrist_aligned_with_face:
                    is_salute_present = True

            # Adjust alpha based on whether salute is present and effect duration
            if is_salute_present:
                # If salute is present, gradually increase alpha until max, then keep at max
                if self.current_effect_frame > 0:
                    self.current_effect_frame -= 1
                alpha = 1.0 - (self.current_effect_frame / self.effect_duration)
                alpha = min(alpha, 1.0) # Cap alpha at 1.0
            else:
                # If salute is not present, gradually decrease alpha
                self.current_effect_frame += 1
                alpha = 1.0 - (self.current_effect_frame / self.effect_duration)
                alpha = max(alpha, 0.0) # Ensure alpha doesn't go below 0.0

            # If alpha reaches 0 and salute is not present, end the effect
            if alpha <= 0 and not is_salute_present:
                 self.is_effect_active = False
                 self.reaction_manager.set_reaction_active(False)
                 self.current_effect_frame = 0 # Reset frame counter

            # Extract the alpha channel from the resized image
            alpha_s = resized_salute_image[:, :, 3] / 255.0 * alpha
            alpha_l = 1.0 - alpha_s

            # Overlay the image onto the frame with gradual alpha blending
            y1, y2 = max(0, y_offset), min(frame_height, y_offset + target_height)
            x1, x2 = max(0, x_offset), min(frame_width, x_offset + target_width)

            img_y1 = max(0, -y_offset)
            img_y2 = img_y1 + (y2 - y1)
            img_x1 = max(0, -x_offset)
            img_x2 = img_x1 + (x2 - x1)

            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s[img_y1:img_y2, img_x1:img_x2] * resized_salute_image[img_y1:img_y2, img_x1:img_x2, c] +
                                          alpha_l[img_y1:img_y2, img_x1:img_x2] * frame[y1:y2, x1:x2, c])