import mediapipe as mp
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import asyncio
import websockets

# Initializing the array for the gestures
SHOW_CAMERA = False
MAX_EXPRESSION_ARRAY_LENGTH = 10
expressions = []
last_expression = None
detection_result = None

# https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index#models
model_path = "face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def is_eye_closed(landmarks, eye_indices):
    upper_lid = landmarks[eye_indices["upper"]]
    lower_lid = landmarks[eye_indices["lower"]]

    vertical_distance = abs(upper_lid.y - lower_lid.y)
    return vertical_distance < 0.02


def check_eyes_closed(face_landmarks):
    left_eye_indices = {"upper": 159, "lower": 145}
    right_eye_indices = {"upper": 386, "lower": 374}

    is_left_eye_closed = is_eye_closed(face_landmarks, left_eye_indices)
    is_right_eye_closed = is_eye_closed(face_landmarks, right_eye_indices)

    return is_left_eye_closed, is_right_eye_closed


def recognize_custom_expression(face_landmarks):
    left_eye_closed, right_eye_closed = check_eyes_closed(face_landmarks)
    if left_eye_closed and right_eye_closed:
        return "EYES_CLOSED"
    elif left_eye_closed and not right_eye_closed:
        return "LEFT_EYE_CLOSED"
    elif right_eye_closed and not left_eye_closed:
        return "RIGHT_EYE_CLOSED"

    return None


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def view_last_expression():
    global last_expression
    print(last_expression)


def save_result(
    result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int  # type: ignore
):
    global detection_result, last_expression, expressions
    detection_result = result
    try:
        last_expression = recognize_custom_expression(
            detection_result.face_landmarks[0]
        )
        if last_expression and last_expression != "None":
            expressions.append(last_expression)
    except:
        last_expression = None

    if len(expressions) > MAX_EXPRESSION_ARRAY_LENGTH:
        expressions.pop(0)


async def send_messages(websocket):
    print("Client connected")
    try:
        while True:
            await websocket.send(f"{last_expression}")
            await asyncio.sleep(0.05)
    except websockets.ConnectionClosed:
        print("Client disconnected")


async def camera_proc():
    global detection_result
    cam = cv2.VideoCapture(0)
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=save_result,
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

                # Uncomment this line for verification if the camera detects signal:
                # view_last_expression()

                if SHOW_CAMERA:
                    cv2.imshow(
                        "Camera", draw_landmarks_on_image(frame, detection_result)
                    )
                    if cv2.waitKey(1) == ord("q"):
                        break

                await asyncio.sleep(0.05)

            except Exception as e:
                print(f"Unhandled exception: {e}")

        cam.release()

        if SHOW_CAMERA:
            cv2.destroyAllWindows()


async def main():
    server = await websockets.serve(send_messages, "localhost", 8765)
    print("Server WebSocket ws://localhost:8765")
    await asyncio.gather(camera_proc(), asyncio.Future())


if __name__ == "__main__":
    asyncio.run(main())
