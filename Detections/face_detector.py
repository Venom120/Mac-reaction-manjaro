import mediapipe as mp
import cv2

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence)

    def detect(self, frame):
        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame and find faces
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                # Extract bounding box information
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                        int(bboxC.width * iw), int(bboxC.height * ih))
                
                # Extract key points (like eyes, nose, mouth, cheeks)
                keypoints = []
                for kp in detection.location_data.relative_keypoints:
                    x = int(kp.x * iw)
                    y = int(kp.y * ih)
                    keypoints.append((x, y))
                    
                faces.append({"bbox": bbox, "keypoints": keypoints})
                
        return faces

    def __del__(self):
        self.face_detection.close()

if __name__ == '__main__':
    # Example usage
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        faces = detector.detect(frame)

        # Draw bounding boxes and keypoints on the frame
        for face in faces:
            bbox = face["bbox"]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
            for kp in face["keypoints"]:
                cv2.circle(frame, kp, 5, (0, 255, 0), -1)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()