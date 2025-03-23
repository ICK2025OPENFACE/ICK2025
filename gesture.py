import time
import mediapipe as mp
import cv2

# Initializing the array for the gestures
gestures = []
last_gesture = None
MAX_GESTURES_ARRAY_LENGTH = 10


# Initializing the recognition model
# https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/index#models
model_path = "gesture_recognizer.task"
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# printing last gesture
def view_last_gesture():
    global last_gesture
    print(last_gesture)


# saving the recognition restult to array for sequential gestures
def save_result(
    result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int
):
    global gestures
    global last_gesture
    try:
        gesture = result.gestures[0][0].category_name
        last_gesture = gesture
        if gesture and gesture != "None":
            gestures.append(gesture)
    except:
        last_gesture = None

    if len(gestures) > MAX_GESTURES_ARRAY_LENGTH:
        gestures.pop(0)


if __name__ == "__main__":

    cam = cv2.VideoCapture(0)
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=save_result,
    )

    show_camera = True

    with GestureRecognizer.create_from_options(options) as recognizer:
        last_recognition_time = 0
        recognition_interval = 0.1

        while True:
            try:
                ret, frame = cam.read()
                if not ret:
                    print("Failed to capture frame. Exiting...")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                current_time = time.time()
                if current_time - last_recognition_time >= recognition_interval:
                    recognizer.recognize_async(
                        mp_image,
                        int(cv2.getTickCount() / cv2.getTickFrequency() * 1000),
                    )
                    last_recognition_time = current_time

                    view_last_gesture()

                if show_camera:
                    cv2.imshow("Camera", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break

            except Exception as e:
                print(f"Unhandled exception: {e}")
        cam.release()
        if show_camera:
            cv2.destroyAllWindows()
