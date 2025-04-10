import math
from typing import List, Dict, Tuple, Union
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark


def is_eye_closed(
    landmarks: List[NormalizedLandmark], eye_indices: Dict[str, int]
) -> bool:
    """
    Determines if an eye is closed based on the distance between the upper and lower eyelids,
    normalized by the size of the face.

    Args:
        landmarks (List[NormalizedLandmark]): A list of normalized facial landmarks.
        eye_indices (Dict[str, int]): A dictionary with keys 'upper' and 'lower', representing
            the indices of the upper and lower eyelid landmarks.

    Returns:
        bool: True if the eye is closed, False otherwise.

    Notes:
        - The function calculates the vertical distance between the upper and lower eyelids.
        - The distance is normalized by the face height to account for variations in face size.
        - A threshold is used to determine if the eye is closed.
    """
    upper_lid = landmarks[eye_indices["upper"]]
    lower_lid = landmarks[eye_indices["lower"]]

    # Points for face height (chin and forehead) used for scaling distances
    chin = landmarks[152]
    forehead = landmarks[10]

    # Calculate face height
    face_height = abs(chin.y - forehead.y)

    # Calculate normalized vertical distance between eyelids
    vertical_distance = abs(upper_lid.y - lower_lid.y) / face_height

    # Threshold for determining if the eye is closed
    THRESHOLD_EYE_CLOSED = 0.05

    return vertical_distance < THRESHOLD_EYE_CLOSED


def check_eyes_closed(landmarks: List[NormalizedLandmark]) -> Tuple[bool, bool]:
    """
    Determines if the left and right eyes are closed based on facial landmarks.

    Args:
        landmarks (List[NormalizedLandmark]): A list of normalized facial landmarks.

    Returns:
        Tuple[bool, bool]: A tuple containing two boolean values:
            - The first value indicates if the left eye is closed (True if closed, False otherwise).
            - The second value indicates if the right eye is closed (True if closed, False otherwise).

    Notes:
        - The function uses specific landmark indices for the left and right eyes.
        - The `is_eye_closed` function is called for each eye.
    """
    left_eye_indices = {"upper": 159, "lower": 145}
    right_eye_indices = {"upper": 386, "lower": 374}

    is_left_eye_closed = is_eye_closed(landmarks, left_eye_indices)
    is_right_eye_closed = is_eye_closed(landmarks, right_eye_indices)

    return is_left_eye_closed, is_right_eye_closed


def detect_smile_and_open_mouth(landmarks: List[NormalizedLandmark]) -> Tuple[bool]:
    """
    Detects whether the mouth is open and whether the person is smiling based on facial landmarks.
    Args:
        landmarks (List[NormalizedLandmark]): A list of normalized facial landmarks, where each landmark
            contains x and y coordinates. The landmarks should include specific points for the lips,
            mouth corners, and cheeks.
    Returns:
        Tuple[bool]: A tuple containing two boolean values:
            - The first value indicates if the mouth is open (True if open, False otherwise).
            - The second value indicates if the person is smiling (True if smiling, False otherwise).
    Notes:
        - The function calculates the vertical distance between the top and bottom lips to determine
          if the mouth is open.
        - The smile detection is based on the ratio of the mouth width to the face width.
        - Thresholds for detecting an open mouth and a smile are defined as constants:
            - `THRESHOLD_OPEN`: Minimum vertical distance between lips to consider the mouth open.
            - `THRESHOLD_SMILE_RATIO`: Minimum ratio of mouth width to face width to consider a smile.
    """
    # Distance lambda calculation
    distance = lambda p1, p2: math.hypot(p1.x - p2.x, p1.y - p2.y)

    # Open mouth and smile thresholds
    THRESHOLD_OPEN = 0.05
    THRESHOLD_SMILE_RATIO = 0.40

   top_lip = landmarks[12]
    bottom_lip = landmarks[14]
    left_mouth = landmarks[307]
    right_mouth = landmarks[77]

    # Points for face width (temples) used for scaling distances
    left_cheek = landmarks[265]
    right_cheek = landmarks[143]

    # Calculating distances
    open_dist = distance(top_lip, bottom_lip)  # Open mouth (vertical distance)
    face_width = distance(left_cheek, right_cheek)  # Face width
    mouth_width = distance(left_mouth, right_mouth)  # Mouth width

    smile_ratio = mouth_width / face_width  # Ratio of mouth width to face width

    # Detection: Open mouth and smile
    mouth_open = open_dist > THRESHOLD_OPEN
    smile = smile_ratio > THRESHOLD_SMILE_RATIO

    return mouth_open, smile


def detect_head_movement(landmarks, center=None):
    """
    Detects head movment by checking if the center of the head moves outside of a box.

    Args:
        landmarks (list): List of normalised landmarks. 
        center (tuple or None): Central position (x, y).

    Returns:
        tuple: (is_left, is_right, is_up, is_down)
    """

    # Center of the face
    face_x = sum([landmark.x for landmark in landmarks]) / len(landmarks)
    face_y = sum([landmark.y for landmark in landmarks]) / len(landmarks)

    # Width of the face
    left_face_x = landmarks[234].x
    right_face_x = landmarks[454].x
    face_width = abs(right_face_x - left_face_x)

    if center is None:
        center = (face_x, face_y)

    center_x, center_y = center

    margin_x = face_width * 0.5
    margin_y = face_width * 0.5

    # Box boundries 
    left_bound = center_x - margin_x
    right_bound = center_x + margin_x
    top_bound = center_y - margin_y
    bottom_bound = center_y + margin_y

    # Movment detecion
    is_left = face_x < left_bound
    is_right = face_x > right_bound
    is_up = face_y < top_bound
    is_down = face_y > bottom_bound

    return (is_left, is_right, is_up, is_down), center
