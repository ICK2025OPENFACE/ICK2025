import mediapipe as mp
import cv2
import faceexpressions as fe
import supportfunctions as sf
import time
import json

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

    if SHOW_CAMERA:
        detection_result = result

    # trying to get signals and in case of unknown error display information about exception
    try:

        # if no face landmarks detected do not pass it to the function to avoid exiting app
        if result.face_landmarks and len(result.face_landmarks) > 0:

            # receiving bool value about eyes test signal
            just_closed, opened_too_fast, activate_action = fe.check_eyes_closed(
                result.face_landmarks[0]
            )
            msg = f"({GROUP_ID})({time.time()})"
            if just_closed:
                msg += f"{face_config["EYE_CHARGING"]}"
            elif opened_too_fast:
                msg += f"{face_config["EYE_FAILED"]}"
            elif activate_action:
                msg += f"{face_config["EYE_ACTIVATION"]}"
            else:
                msg = None  # None - msg won't be send due to send_msg_via_udp implementation

            # sending a int value to game server to handle corresponding signal
            sf.send_msg_via_udp(msg, SERVER_IP, SERVER_PORT)

            # receiving bool value about mouth
            opened_mouth, smile = fe.detect_smile_and_open_mouth(
                result.face_landmarks[0]
            )
            if opened_mouth:
                msg = f"({GROUP_ID})({time.time()})({face_config["MOUTH_OPENED"]})"
                sf.send_msg_via_udp(msg, SERVER_IP, SERVER_PORT)
            if smile:
                msg = f"({GROUP_ID})({time.time()})({face_config["SMILE"]})"
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
