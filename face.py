import mediapipe as mp
import cv2
import faceexpressions as fe
import supportfunctions as sf

# Networking setup for UDP
server_ip = "127.0.0.1"
server_port = 4242

# Additional visualization parameter
SHOW_CAMERA = False
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
    Function is a callback for mediapipe FaceLandmarker model.
    It calls functions from faceexpressions.py that returns information
    about current signals detected from user face and send it to game udp server.
    @params:
        result: FaceLandmarkerResult - is a result of model face landmarks detection.
        It contains object with property face_landmarks that contains list of NormalizedLandmark
        which are used for further proceeding and signal generations.
        output_image: mp.Image - is a default parameter that needs to be passed for FaceLandmarkerResult work.
        timestamp_ms:int - is a default parameter that needs to be passed for FaceLandmarkerResult work.

    @output:
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

            # receiving int value about eyes test signal
            test_expression = fe.eyes_expression(result.face_landmarks[0])

            # sending a int value to game server to handle corresponding signal
            sf.send_msg_via_udp(test_expression, server_ip, server_port)

    except Exception as e:
        print(f"Unhandled exception in camera_callback function: {e}")


def camera_proc():
    """
    Main script function that will be called when
    merging solutions with other modalities and a game.

    @params:
        None

    @output:
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
