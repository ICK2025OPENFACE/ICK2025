import mediapipe as mp
import cv2
import socket
import faceexpressions as fe
import supportfunctions as sf

# Networking setup for UDP
server_ip = "127.0.0.1"
server_port = 4242

# Additional visualization parameter
SHOW_CAMERA = True
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
):
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
    global detection_result
    if SHOW_CAMERA:
        detection_result = result

    try:
        if result.face_landmarks and len(result.face_landmarks) > 0:
            test_expression = fe.eyes_expression(result.face_landmarks[0])
            sf.send_msg_via_udp(test_expression, server_ip, server_port)
    except Exception as e:
        print(f"Unhandled exception in camera_callback function: {e}")


def camera_proc():
    cam = cv2.VideoCapture(0)
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=camera_callback,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            try:
                ret, frame = cam.read()
                if not ret:
                    print("Failed to capture frame. Exiting...")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                landmarker.detect_async(
                    mp_image,
                    int(cv2.getTickCount() / cv2.getTickFrequency() * 1000),
                )

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
    camera_proc()
