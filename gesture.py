import time
import mediapipe as mp
import cv2

gestures = []
# https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/index#models
model_path = "gesture_recognizer.task"
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(
    result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int
):
    global gestures
    try:
        gesture = result.gestures[0][0].category_name
        if gesture and gesture != "None":
            gestures.append(gesture)
            print("Recognized Gesture:", gestures[-1])
    except:
        pass

    if len(gestures) > 10:
        gestures.pop(0)


cam = cv2.VideoCapture(0)
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)

with GestureRecognizer.create_from_options(options) as recognizer:
    last_recognition_time = 0
    recognition_interval = 0.1

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        current_time = time.time()
        if current_time - last_recognition_time >= recognition_interval:
            recognizer.recognize_async(
                mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            )
            last_recognition_time = current_time

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
