import mediapipe as mp
import cv2
import faceexpressions as fe
import supportfunctions as sf
import time
import json
import os
import requests

if not os.path.isfile("face_config.json"):
    print("Missing config file, download face_config.json before running")
    quit()

if not os.path.isfile("face_landmarker.task"):
    file_name = "face_landmarker.task"
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    print(f"Missing {file_name}; Downloading from: ")
    print(url)
    response = requests.get(url)
    with open(file_name, "wb") as file:
        file.write(response.content)
    print("Download completed")

# Reading settings
with open("face_config.json", "r") as file:
    face_config = json.load(file)

# Networking setup for UDP
SERVER_IP = face_config["SERVER_IP"]
SERVER_PORT = face_config["SERVER_PORT"]
GROUP_ID = face_config["GROUP_ID"]

# Additional visualization parameter
SHOW_CAMERA = face_config["SHOW_CAMERA"]
detection_result = None
center = None

# https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index#models
model_path = "face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def camera_callback(
    result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int  # type: ignore
) -> None:
    """
    Callback function for the MediaPipe FaceLandmarker model.
    This function processes the detected face landmarks, analyzes facial expressions,
    and sends corresponding signals to a game server via UDP.

    Args:
        result (FaceLandmarkerResult): The result of the face landmarks detection.
            Contains a property `face_landmarks` which is a list of `NormalizedLandmark`
            objects used for further processing and signal generation.
        output_image (mp.Image): A default parameter required for FaceLandmarkerResult processing.
        timestamp_ms (int): A default parameter required for FaceLandmarkerResult processing.

    Returns:
        None
    """

    # in case visualization is necessary detection_result will be passed to draw_landmarks_on_image
    global detection_result
    global center

    if SHOW_CAMERA:
        detection_result = result

    if result is None:
        return
    # trying to get signals and in case of unknown error display information about exception
    try:

        # if no face landmarks detected do not pass it to the function to avoid exiting app
        if result.face_landmarks and len(result.face_landmarks) > 0:
            if face_config["BOOLEAN_MSG"]:
                # creating container for bool message
                signals = dict()

                # receiving bool value about eyes test signal
                test_left_eye, test_right_eye = fe.check_eyes_closed(
                    result.face_landmarks[0]
                )

                # receiving bool value about mouth
                opened_mouth, smile = fe.detect_smile_and_open_mouth(
                    result.face_landmarks[0]
                )

                # receiving bool value about face movement
                (is_left, is_right, is_up, is_down), center = fe.detect_head_movement(
                    result.face_landmarks[0], center
                )

                signals[face_config["LEFT_EYE_CLOSED"]] = test_left_eye
                signals[face_config["RIGHT_EYE_CLOSED"]] = test_right_eye
                signals[face_config["BOTH_EYES_CLOSED"]] = (
                    test_left_eye and test_right_eye
                )
                signals[face_config["MOUTH_OPENED"]] = opened_mouth
                signals[face_config["SMILE"]] = smile
                signals[face_config["IS_LEFT"]] = is_left
                signals[face_config["IS_RIGHT"]] = is_right
                signals[face_config["IS_UP"]] = is_up
                signals[face_config["IS_DOWN"]] = is_down

                signals = dict(sorted(signals.items()))

                msg = f"{face_config["GROUP_ID"]}"
                msg += "".join(str(int(value)) for value in signals.values())
                # sending a boolean values to game server to handle corresponding signal
                sf.send_msg_via_udp(msg, SERVER_IP, SERVER_PORT)

            else:
                if True:
                    # receiving bool value about eyes test signal
                    test_left_eye, test_right_eye = fe.check_eyes_closed(
                        result.face_landmarks[0]
                    )
                    msg = f"({GROUP_ID})({time.time()})"
                    if test_right_eye and test_left_eye:
                        msg += f"{face_config["BOTH_EYES_CLOSED"]}"
                    elif test_left_eye:
                        msg += f"{face_config["LEFT_EYE_CLOSED"]}"
                    elif test_right_eye:
                        msg += f"{face_config["RIGHT_EYE_CLOSED"]}"
                    else:
                        msg = None  # None - msg won't be send due to send_msg_via_udp implementation

                    # sending a int value to game server to handle corresponding signal
                    sf.send_msg_via_udp(msg, SERVER_IP, SERVER_PORT)

                if True:
                    # receiving bool value about mouth
                    opened_mouth, smile = fe.detect_smile_and_open_mouth(
                        result.face_landmarks[0]
                    )

                    if opened_mouth:
                        msg = (
                            f"({GROUP_ID})({time.time()}){face_config["MOUTH_OPENED"]}"
                        )
                        # sending a int value to game server to handle corresponding signal
                        sf.send_msg_via_udp(msg, SERVER_IP, SERVER_PORT)

                    if smile:
                        msg = f"({GROUP_ID})({time.time()}){face_config["SMILE"]}"
                        # sending a int value to game server to handle corresponding signal
                        sf.send_msg_via_udp(msg, SERVER_IP, SERVER_PORT)

                if True:
                    # receiving bool value about face movement
                    (is_left, is_right, is_up, is_down), center = (
                        fe.detect_head_movement(result.face_landmarks[0], center)
                    )
                    msg = f"({GROUP_ID})({time.time()})"
                    if is_left:
                        msg += f"{face_config["IS_LEFT"]}"
                        # sending a int value to game server to handle corresponding signal
                        sf.send_msg_via_udp(msg, SERVER_IP, SERVER_PORT)
                    if is_right:
                        msg += f"{face_config["IS_RIGHT"]}"
                        # sending a int value to game server to handle corresponding signal
                        sf.send_msg_via_udp(msg, SERVER_IP, SERVER_PORT)
                    if is_up:
                        msg += f"{face_config["IS_UP"]}"
                        # sending a int value to game server to handle corresponding signal
                        sf.send_msg_via_udp(msg, SERVER_IP, SERVER_PORT)
                    if is_down:
                        msg += f"{face_config["IS_DOWN"]}"
                        # sending a int value to game server to handle corresponding signal
                        sf.send_msg_via_udp(msg, SERVER_IP, SERVER_PORT)

    except Exception as e:
        print(f"Unhandled exception in camera_callback function: {e}")


def camera_proc():
    """
    Main script function that initializes the camera processing pipeline.
    This function sets up the camera feed, configures the FaceLandmarker model,
    and processes the video stream to detect and analyze facial expressions.

    Args:
        None

    Returns:
        None
    """

    # grabbing the camera output
    cam = cv2.VideoCapture(0)

    # initializing FaceLandmarker model options
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=camera_callback,
    )

    # creating a main loop with model as a landmarker object
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            try:
                # receiving frames from camera
                ret, frame = cam.read()

                if not ret:
                    print("Failed to capture frame. Exiting...")
                    break

                # parsing BGR to RGB due to model standard
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # parsing rbg frame into mp.Image object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # detection landmarks on given frame (as mp.Image object)
                landmarker.detect_async(
                    mp_image,
                    int(cv2.getTickCount() / cv2.getTickFrequency() * 1000),
                )

                # displaying the script output with landmarks if SHOW_CAMERA set to true
                if detection_result is None:
                    continue
                if SHOW_CAMERA:
                    cv2.imshow(
                        "Camera", sf.draw_landmarks_on_image(frame, detection_result)
                    )
                    if cv2.waitKey(1) == ord("q"):
                        break

            except Exception as e:
                print(f"Unhandled exception: {e}")

        cam.release()

        if SHOW_CAMERA:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # executing main function of script
    camera_proc()
