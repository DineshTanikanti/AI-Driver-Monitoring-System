import cv2
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self):
        # This is the standard entry point for all 0.10+ versions
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            # Return list of (x, y) coordinates for all 468 points
            return [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
        return None