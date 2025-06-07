import cv2
import mediapipe as mp
import random

mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

class Emoji:
    def __init__(self, x, y, vx, vy, scale, image_path):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.scale = scale
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.image is not None:
            self.image = cv2.resize(self.image, (int(50 * scale), int(50 * scale)))

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.5  # gravity

    def draw(self, frame):
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        x1 = int(self.x)
        y1 = int(self.y)
        x2 = x1 + w
        y2 = y1 + h

        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return

        overlay = self.image[:, :, :3]
        mask = self.image[:, :, 3:] / 255.0

        roi = frame[y1:y2, x1:x2]
        for c in range(3):
            roi[:, :, c] = (1 - mask[:, :, 0]) * roi[:, :, c] + mask[:, :, 0] * overlay[:, :, c]
        frame[y1:y2, x1:x2] = roi


from Detections.thumbs_up_detector import ThumbsUpDetector
from Detections.peace_detector import PeaceDetector
from Detections.heart_detector import HeartDetector
from Detections.blush_detector import BlushDetector
from Detections.fist_bump_detector import FistBumpDetector
from Detections.salute_detector import SaluteDetector # Import the new detector
from reaction_manager import ReactionManager # Import the new class
mp_drawing = None
# mp_drawing = mp.solutions.drawing_utils

class Reactions:
    def __init__(self):
        # Initialize MediaPipe Hands and Face Detection
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.9)
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
        self.active_emojis = []
        self.reaction_manager = ReactionManager() # Instantiate ReactionManager
        self.thumbs_up_detector = ThumbsUpDetector(self, self.reaction_manager) # Pass manager
        self.peace_detector = PeaceDetector(self, self.reaction_manager) # Pass manager
        self.heart_detector = HeartDetector(self, self.reaction_manager) # Pass manager
        self.blush_detector = BlushDetector(self, self.reaction_manager) # Pass manager
        self.fist_bump_detector = FistBumpDetector(self, self.reaction_manager) # Pass manager
        self.salute_detector = SaluteDetector(self, self.reaction_manager) # Instantiate the new detector


    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process for hands
        hand_results = self.hands.process(rgb)
        hands = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None

        # Process for faces
        face_results = self.face_detection.process(rgb)
        faces = face_results.detections if face_results.detections else None

        # Draw hand landmarks
        if hands and mp_drawing:
            for hand_landmarks in hands:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        # # Draw face detections
        if mp_drawing:
            if faces and mp_drawing:
                for detection in faces:
                    mp_drawing.draw_detection(frame, detection)


        # Only run detectors if no reaction is currently active
        if not self.reaction_manager.is_reaction_active():
            self.thumbs_up_detector.detect(hands, frame)
            self.peace_detector.detect(hands, frame)
            if hands and len(hands) == 2:
                self.heart_detector.detect(hands, frame)
                self.blush_detector.detect(hands, frame)
                self.fist_bump_detector.detect(hands, frame)
            self.salute_detector.detect(hands, frame) # SaluteDetector now handles its own face detection


        # Always apply effects, as they manage their own duration
        # Pass the detected hand landmarks and face detections to apply_effect methods
        self.thumbs_up_detector.apply_effect(frame, hands)
        self.peace_detector.apply_effect(frame, hands)
        self.heart_detector.apply_effect(frame, hands)
        self.blush_detector.apply_effect(frame, hands)
        self.fist_bump_detector.apply_effect(frame, hands)
        self.salute_detector.apply_effect(frame, hands, faces) # Pass face detections to salute effect

        self.update_and_draw_emojis(frame)

    def spawn_fountain_emojis(self, frame, image_path):
        h, w = frame.shape[:2]
        cx = w // 2
        cy = h - 100
        vx = random.uniform(-2, 2)
        vy = random.uniform(-10, -7) # Increased upward velocity
        scale = random.uniform(0.4, 0.6)  # Scale factor for emoji size
        self.active_emojis.append(Emoji(cx, cy, vx, vy, scale, image_path))

    def update_and_draw_emojis(self, frame):
        updated = []
        for emoji in self.active_emojis:
            emoji.update()
            emoji.draw(frame)
            if -50 < emoji.y < frame.shape[0] + 50:
                updated.append(emoji)
        self.active_emojis = updated
