import cv2
import numpy as np

def get_head_angles(frame, landmarks):
    try:
        # Standard 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        image_points = np.array([
            landmarks[1],   # Nose tip
            landmarks[152], # Chin
            landmarks[33],  # Left eye corner
            landmarks[263], # Right eye corner
            landmarks[61],  # Left mouth corner
            landmarks[291]  # Right mouth corner
        ], dtype="double")

        size = frame.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]], 
                                 [0, focal_length, center[1]], 
                                 [0, 0, 1]], dtype="double")

        dist_coeffs = np.zeros((4,1))
        success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        
        if success:
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            return angles[0], angles[1] # Pitch, Yaw
            
    except Exception as e:
        print(f"Head Pose Error: {e}")
        
    return None # Return None if calculation fails