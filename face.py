import mediapipe as mp
import cv2
import asyncio
import websockets
import faceexpressions as fe
import support_functions as sf

# Initializing the array for the gestures
SHOW_CAMERA = True
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


# TODO usunąć funkcję zapisu "ostatnich ekspresji" i aktulanej ekspresji,
# zamiast tego przesyłać info zgodne z tableką na teams
def save_result(
    result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int  # type: ignore
):
    global detection_result, last_expression, expressions
    detection_result = result
    try:
        last_expression = fe.eyes_expression(detection_result.face_landmarks[0])
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

                if SHOW_CAMERA:
                    cv2.imshow(
                        "Camera", sf.draw_landmarks_on_image(frame, detection_result)
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
