import cv2
import numpy as np
from vision.face_mesh import FaceMeshDetector
from vision.head_pose import get_head_angles
from vision.ear import calculate_ear
from model.eye_predictor import predict_eye
from logic.fatigue_logic import FatigueDetector

L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)
face_mesh = FaceMeshDetector()
fatigue = FatigueDetector()

frame_count = 0
cnn_val = 0.0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    landmarks = face_mesh.get_landmarks(frame)
    if landmarks:
        # 1. Faster EAR every frame
        ear_l = calculate_ear([landmarks[i] for i in L_EYE])
        ear_r = calculate_ear([landmarks[i] for i in R_EYE])
        avg_ear = (ear_l + ear_r) / 2.0

        # 2. Run heavy CNN only every 3rd frame to save CPU (Increases FPS)
        if frame_count % 3 == 0:
            x, y, w, h = cv2.boundingRect(np.array([landmarks[i] for i in L_EYE]))
            eye_crop = frame[y:y+h, x:x+w]
            cnn_val = predict_eye(eye_crop)

        # 3. Head Pose
        pitch, yaw = get_head_angles(frame, landmarks)

        # 4. Multi-Factor Fusion
        status, color = fatigue.evaluate(cnn_val, avg_ear, pitch)

        # 5. UI
        cv2.putText(frame, status, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        # Optional: Show a red alert if head is turned away (Distraction)
        if abs(yaw) > 25:
            cv2.putText(frame, "DISTRACTED", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("AI Driver Fatigue Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()