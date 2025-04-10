import math
from typing import List, Dict, Tuple, Union
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import time

CLOSED_TRESH = 0.1 # maximal tolerable eye gap
BUF_SIZE = 10
CLOSED_TIME = 3 #[s]
MAX_BLINK_DURATION = 0.5 #[s]

def euclideanDistance(pointA: NormalizedLandmark, pointB: NormalizedLandmark) -> float:
    """
    Returns the Euclidean distance between two Landmarks.
    Args:
        pointA (NormalizedLandmark): first point.
        pointB (NormalizedLandmark): second point.
    Returns:
        float: distance.
    """

    p = (pointA.x, pointA.y, pointA.z)
    q = (pointB.x, pointB.y, pointB.z)
    distance = math.dist(p,q)
    return distance


def is_eye_closed(landmarks: List[NormalizedLandmark], eye_indices: Dict[str, int] ) -> float:
    """
    Determines if an eye is closed based on the distance between the upper and lower eyelids,
    normalized by the width of the eye.

    Args:
        landmarks (List[NormalizedLandmark]): A list of normalized facial landmarks.
        eye_indices (Dict[str, int]): A dictionary with keys 'h1', 'h2', 'v1' and 'v2', representing
            the indices of the horizontal and vertical eye landmarks.

    Returns:
        float: Eye open coefficient.

    Notes:
        - The function calculates the vertical distance between the upper and lower eyelids.
        - The distance is normalized by the eye width.
    """
    h1 = landmarks[eye_indices["h1"]]
    h2 = landmarks[eye_indices["h2"]]
    v1 = landmarks[eye_indices["v1"]]
    v2 = landmarks[eye_indices["v2"]]

    horizontal_distance = euclideanDistance(h1, h2)
    vertical_distance = euclideanDistance(v1, v2)
    ratio = vertical_distance / horizontal_distance
    return ratio


def check_eyes_closed(landmarks: List[NormalizedLandmark]) -> Tuple[bool, bool, bool]:
    """
    Determines if the left and right eyes are closed based on facial landmarks.

    Args:
        landmarks (List[NormalizedLandmark]): A list of normalized facial landmarks.

    Returns:
        bool: Pulse when eyes have been closed.
        bool: Pulse when eyes opened before CLOSED_TIME.
        bool: Eyes have been closed for predefined time.

    Notes:
        - The function uses specific landmark indices for the left and right eyes.
        - The `is_eye_closed` function is called for each eye.
    """


    # Initialize memory attributes once
    if not hasattr(check_eyes_closed, "in_closed"):
        check_eyes_closed.in_closed = False
        check_eyes_closed.last_trigger_t = 0
        check_eyes_closed.output_triggered = False
        check_eyes_closed.was_activated = False
        check_eyes_closed.valid_closure = False
        check_eyes_closed.states_buf = {"left": [], "right": []}


    left_eye_indices = {"h1": 362, "h2": 263, "v1": 386, "v2": 374}
    right_eye_indices = {"h1": 33, "h2": 374, "v1": 159, "v2": 145}

    left_ratio = is_eye_closed(landmarks, left_eye_indices)
    right_ratio = is_eye_closed(landmarks, right_eye_indices)

    # create moving buffer
    check_eyes_closed.states_buf["left"].append(left_ratio)
    check_eyes_closed.states_buf["right"].append(right_ratio)
    if len(check_eyes_closed.states_buf["left"]) > BUF_SIZE:
        check_eyes_closed.states_buf["left"].pop(0)
        check_eyes_closed.states_buf["right"].pop(0)

    # calculate averages
    avg_left = sum(check_eyes_closed.states_buf["left"]) / len(check_eyes_closed.states_buf["left"])
    avg_right = sum(check_eyes_closed.states_buf["right"]) / len(check_eyes_closed.states_buf["right"])


    # decide if eyes are closed and then save current time
    current_t = time.time()

    eyes_closed_output = False
    eyes_failed = False
    activate = False


    # Eyes currently closed
    if avg_left < CLOSED_TRESH and avg_right < CLOSED_TRESH:
        if not check_eyes_closed.in_closed:
            # Eyes just closed — start timing
            check_eyes_closed.in_closed = True
            check_eyes_closed.last_trigger_t = current_t
            check_eyes_closed.output_triggered = False
            check_eyes_closed.was_activated = False
            check_eyes_closed.valid_closure = False  # wait to see if it's not a blink

        # Check if closure passed blink threshold
        if not check_eyes_closed.valid_closure:
            if current_t - check_eyes_closed.last_trigger_t >= MAX_BLINK_DURATION:
                eyes_closed_output = True  # Only pulse once when valid closure confirmed
                check_eyes_closed.valid_closure = True

        # Activate if eyes have stayed closed long enough
        if (current_t - check_eyes_closed.last_trigger_t >= CLOSED_TIME
                and not check_eyes_closed.output_triggered):
            activate = True
            check_eyes_closed.output_triggered = True
            check_eyes_closed.was_activated = True

    # Eyes currently open
    else:
        if check_eyes_closed.in_closed:
            # Only fail if it was a valid (non-blink) closure and no activation happened
            if check_eyes_closed.valid_closure and not check_eyes_closed.was_activated:
                eyes_failed = True

        # Reset state
        check_eyes_closed.in_closed = False
        check_eyes_closed.output_triggered = False
        check_eyes_closed.was_activated = False
        check_eyes_closed.valid_closure = False


    return eyes_closed_output, eyes_failed, activate



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

    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]

    # Points for face width (temples) used for scaling distances
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]

    # Calculating distances
    open_dist = distance(top_lip, bottom_lip)  # Open mouth (vertical distance)
    face_width = distance(left_cheek, right_cheek)  # Face width
    mouth_width = distance(left_mouth, right_mouth)  # Mouth width

    smile_ratio = mouth_width / face_width  # Ratio of mouth width to face width

    # Detection: Open mouth and smile
    mouth_open = open_dist > THRESHOLD_OPEN
    smile = smile_ratio > THRESHOLD_SMILE_RATIO

    return mouth_open, smile
