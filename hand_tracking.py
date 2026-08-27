import cv2
import mediapipe as mp
import time
import serial

# -----------------------------
# ARDUINO CONNECTION
# -----------------------------

arduino = serial.Serial("COM3", 115200)
time.sleep(2)

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# -----------------------------
# CAMERA
# -----------------------------

camera = cv2.VideoCapture(0)

previous_timestamp = 0
finger_was_up = False

while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp = int(time.monotonic() * 1000)

    if timestamp <= previous_timestamp:
        timestamp = previous_timestamp + 1

    previous_timestamp = timestamp

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        height, width, _ = frame.shape

        # Index fingertip = landmark 8
        # Index middle joint = landmark 6
        finger_is_up = hand[8].y < hand[6].y

        if finger_is_up:

            cv2.putText(
                frame,
                "INDEX FINGER UP",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Only send A when finger changes
            # from DOWN -> UP
            if not finger_was_up:
                arduino.write(b"A")
                print("Sent A to Arduino")
                finger_was_up = True

        else:
            finger_was_up = False

        # Draw hand landmarks
        for landmark in hand:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

    else:
        finger_was_up = False

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
arduino.close()
landmarker.close()
cv2.destroyAllWindows()