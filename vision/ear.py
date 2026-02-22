import numpy as np  # <--- Added this

def calculate_ear(eye_pts):
    # Vertical distances
    A = np.linalg.norm(np.array(eye_pts[1]) - np.array(eye_pts[5]))
    B = np.linalg.norm(np.array(eye_pts[2]) - np.array(eye_pts[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(eye_pts[0]) - np.array(eye_pts[3]))
    return (A + B) / (2.0 * C)