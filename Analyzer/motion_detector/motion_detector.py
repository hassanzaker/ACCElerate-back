import numpy as np
from scipy.spatial import distance

class MotionDetector():
    def analyze_head_motion(keypoints, prev_keypoints, time_delta):
        """
        Analyze head motion to detect scanning patterns.
        :param keypoints: Current keypoints array [num_keypoints, 3].
        :param prev_keypoints: Previous keypoints array [num_keypoints, 3].
        :param time_delta: Time difference between frames (in seconds).
        :return: Motion type (e.g., "scanning_horizontal", "scanning_vertical", None).
        """
        if keypoints is None or prev_keypoints is None:
            return None  # Not enough data to analyze

        # Extract head-related keypoints
        nose = keypoints[0][:2]
        left_eye = keypoints[1][:2]
        right_eye = keypoints[2][:2]
        left_ear = keypoints[3][:2]
        right_ear = keypoints[4][:2]
        print("============")
        print(nose, left_eye, right_eye, left_ear, right_ear)

        # Previous head-related keypoints
        prev_nose = prev_keypoints[0][:2]
        prev_left_eye = prev_keypoints[1][:2]
        prev_right_eye = prev_keypoints[2][:2]


        # Calculate horizontal and vertical displacements
        horizontal_displacement = np.abs(nose[0] - prev_nose[0])
        vertical_displacement = np.abs(nose[1] - prev_nose[1])

        print("HV", horizontal_displacement, vertical_displacement)
        print("*************************")
        # Detect scanning patterns
        if horizontal_displacement > 20 and vertical_displacement < 10:
            return "scanning_horizontal"  # Side-to-side scanning
        elif vertical_displacement > 20 and horizontal_displacement < 10:
            return "scanning_vertical"  # Up-and-down scanning

        return None  # No scanning detected