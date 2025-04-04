from typing import List, Dict, Tuple, Union
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark


# TODO vertical_distance variable should be compared with normalized value (calculated from size of face), not constant
def is_eye_closed(
    landmarks: List[NormalizedLandmark], eye_indices: Dict[str, int]
) -> bool:
    """
    Function returns the boolean value that determinate if eye is closed.
    The distance between upper lid and lower lid is calculated based on the face size.
    @params:
        landmarks: List[NormalizedLandmark] - given list of face landmarks points.
        eye_indices: Dict[AnyStr, int] - dictionary with keys 'upper' and 'lower'
        and it's points numbers corresponding to face landmarks points.

    @output:
        boolean value determining if eye is closed.
    """
    upper_lid = landmarks[eye_indices["upper"]]
    lower_lid = landmarks[eye_indices["lower"]]

    vertical_distance = abs(upper_lid.y - lower_lid.y)
    return vertical_distance < 0.02


def check_eyes_closed(landmarks: List[NormalizedLandmark]) -> Tuple[bool]:
    """
    Function returns the tuple of boolean values, both of them determinate specific
    (left, right) eye is closed.
    @params:
        landmarks: List[NormalizedLandmark] - given list of face landmarks points.

     @output:
        tuple of boolean value determining if (left, right) eye is closed.
    """
    left_eye_indices = {"upper": 159, "lower": 145}
    right_eye_indices = {"upper": 386, "lower": 374}

    is_left_eye_closed = is_eye_closed(landmarks, left_eye_indices)
    is_right_eye_closed = is_eye_closed(landmarks, right_eye_indices)

    return is_left_eye_closed, is_right_eye_closed


def eyes_expression(landmarks: List[NormalizedLandmark]) -> Union[None, str]:
    """
    Function returns the string natural language value that determine which eye is closed:
    'LEFT_EYE_CLOSED' - when only left eye is closed;
    'RIGHT_EYE_CLOSED' - when only right eye is closed;
    'EYES_CLOSED' - when both eyes are closed;
    None - when eyes are opened.
    @params:
        landmarks: List[NormalizedLandmark] - given list of face landmarks points.

     @output:
        string value None/EYES_CLOSED/LEFT_EYE_CLOSED/RIGHT_EYE_CLOSED.
    """
    left_eye_closed, right_eye_closed = check_eyes_closed(landmarks)
    if left_eye_closed and right_eye_closed:
        return "EYES_CLOSED"
    elif left_eye_closed and not right_eye_closed:
        return "LEFT_EYE_CLOSED"
    elif right_eye_closed and not left_eye_closed:
        return "RIGHT_EYE_CLOSED"

    return None
