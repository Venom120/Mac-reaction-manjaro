import mediapipe as mp
from .base_detector import BaseDetector
import cv2, random
import numpy as np

mp_hands = mp.solutions.hands

class FistBumpDetector(BaseDetector):
    def __init__(self, emoji_spawner, reaction_manager):
        super().__init__(emoji_spawner)
        self.reaction_manager = reaction_manager # Store the ReactionManager instance
        self.fist_bump_detected = False
        self.effect_duration = 15 # frames (Increased speed further)
        self.current_effect_frame = 0
        self.smoke_particles = []
        self.smoke_spawned_for_current_detection = False # Flag to ensure smoke spawns only once per detection

    def detect(self, hands, frame):
        # If another reaction is active, do not perform detection
        if self.reaction_manager.is_reaction_active():
            return

        # Check for two hands
        if not hands or len(hands) != 2:
            self.fist_bump_detected = False
            self.smoke_spawned_for_current_detection = False # Reset flag when hands are not detected
            self.fist_bump_detected = False
            self.smoke_spawned_for_current_detection = False # Reset flag when hands are not detected
            return

        # Ensure hand 1 is the left hand (lower x-value) and hand 2 is the right hand (higher x-value)
        # We'll use the wrist landmark for comparison
        wrist1_x = hands[0].landmark[mp_hands.HandLandmark.WRIST].x
        wrist2_x = hands[1].landmark[mp_hands.HandLandmark.WRIST].x

        if wrist1_x < wrist2_x:
            lm1 = hands[0].landmark # Hand 1 is left
            lm2 = hands[1].landmark # Hand 2 is right
        else:
            lm1 = hands[1].landmark # Hand 1 is left
            lm2 = hands[0].landmark # Hand 2 is right


        # Simple check: are the wrists close to each other?
        wrist1 = lm1[mp_hands.HandLandmark.WRIST]
        wrist2 = lm2[mp_hands.HandLandmark.WRIST]

        distance = np.sqrt((wrist1.x - wrist2.x)**2 + (wrist1.y - wrist2.y)**2)

        # Check if both hands are in a fist shape (simplified check)
        # Check if the distance between wrist and index finger tip is small
        index_tip1 = lm1[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_tip2 = lm2[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        fist_threshold = 0.1 # Adjust based on testing

        is_fist1 = np.sqrt((wrist1.x - index_tip1.x)**2 + (wrist1.y - index_tip1.y)**2) < fist_threshold
        is_fist2 = np.sqrt((wrist2.x - index_tip2.x)**2 + (wrist2.y - index_tip2.y)**2) < fist_threshold


        # Check if wrists are close and both hands are fists
        if distance < 0.3 and is_fist1 and is_fist2: # Adjust distance threshold based on testing
            if not self.fist_bump_detected:
                print("FistBumpDetector: Fist Bump gesture detected!")
                self.fist_bump_detected = True
                self.current_effect_frame = self.effect_duration
                self.smoke_spawned_for_current_detection = False # Reset smoke flag for new detection
                self.reaction_manager.set_reaction_active(True) # Set reaction active when detected
        else:
            self.fist_bump_detected = False
            self.smoke_spawned_for_current_detection = False # Reset flag when gesture is not detected
            self.fist_bump_detected = False

    def apply_effect(self, frame, multi_hand_landmarks):
        if self.current_effect_frame > 0:
            h, w, _ = frame.shape
            progress = (self.effect_duration - self.current_effect_frame) / self.effect_duration

            # Calculate the new width for the squeezed frame (for sideways squeeze)
            squeezed_width = int(w * (1 - progress))
            if squeezed_width < 1: # Prevent width from becoming zero or negative
                squeezed_width = 1

            # Resize the frame
            squeezed_frame = cv2.resize(frame, (squeezed_width, h))

            # Create a black background of the original size
            black_background = np.zeros((h, w, 3), dtype=np.uint8)

            # Calculate the position to place the squeezed frame in the center (horizontally)
            start_x = (w - squeezed_width) // 2
            end_x = start_x + squeezed_width

            # Place the squeezed frame onto the black background
            # Ensure the slice dimensions match the squeezed_frame dimensions
            black_background[:, start_x:end_x] = squeezed_frame[:, :]


            # Copy the result back to the original frame
            frame[:] = black_background[:]

            self.current_effect_frame -= 1

            # If the effect just finished, set reaction inactive
            if self.current_effect_frame == 0:
                 self.reaction_manager.set_reaction_active(False)


        # Spawn smoke after the collapse is complete (when current_effect_frame is 0)
        # and only if smoke hasn't been spawned for this detection yet
        if self.current_effect_frame == 0 and self.fist_bump_detected and not self.smoke_spawned_for_current_detection:
             # Find the center point between the hands to spawn smoke
            if multi_hand_landmarks and len(multi_hand_landmarks) == 2:
                lm1 = multi_hand_landmarks[0].landmark
                lm2 = multi_hand_landmarks[1].landmark
                wrist1 = lm1[mp_hands.HandLandmark.WRIST]
                wrist2 = lm2[mp_hands.HandLandmark.WRIST]
                self.spawn_smoke(frame, wrist1, wrist2)
                self.smoke_spawned_for_current_detection = True # Set flag after spawning smoke
            # Note: self.fist_bump_detected is reset in the detect method when hands are no longer detected or gesture is lost.

        self.update_and_draw_smoke(frame)
        self.update_and_draw_smoke(frame)
    def spawn_smoke(self, frame, wrist1, wrist2):
        h, w, _ = frame.shape
        # Convert normalized coordinates to pixel coordinates
        cx1, cy1 = int(wrist1.x * w), int(wrist1.y * h)
        cx2, cy2 = int(wrist2.x * w), int(wrist2.y * h)

        # Spawn smoke particles around the point between the wrists
        spawn_x = (cx1 + cx2) // 2
        spawn_y = (cy1 + cy2) // 2

        for _ in range(50): # Spawn 50 particles
            self.smoke_particles.append(SmokeParticle(spawn_x, spawn_y))

    def update_and_draw_smoke(self, frame):
        updated_particles = []
        for particle in self.smoke_particles:
            particle.update()
            particle.draw(frame)
            # Keep particles that are still visible and have alpha > 0
            if 0 < particle.x < frame.shape[1] and 0 < particle.y < frame.shape[0] and particle.alpha > 0:
                updated_particles.append(particle)
        self.smoke_particles = updated_particles


class SmokeParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-5, 5)
        self.alpha = 255 # Start fully opaque
        self.decay_rate = random.uniform(2, 10) # How fast alpha decreases
        self.size = random.uniform(4, 7) # Size of the particle

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.alpha -= self.decay_rate
        if self.alpha < 0:
            self.alpha = 0

    def draw(self, frame):
        if self.alpha > 0:
            overlay = frame.copy()
            # Draw a circle for the smoke particle
            cv2.circle(overlay, (int(self.x), int(self.y)), int(self.size), (150, 150, 150), -1) # Grey color
            # Blend the circle with the frame based on alpha
            alpha_channel = self.alpha / 255.0
            cv2.addWeighted(overlay, alpha_channel, frame, 1 - alpha_channel, 0, frame)