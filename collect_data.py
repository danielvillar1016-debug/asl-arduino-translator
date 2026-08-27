import cv2
import mediapipe as mp
import time
import csv

DATA_FILE = "asl_landmarks.csv"

SAMPLES_PER_LETTER = 50
SAMPLE_INTERVAL = 0.07


# --------------------------------
# MEDIAPIPE SETUP
# --------------------------------

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


# --------------------------------
# NORMALIZE LANDMARKS
# --------------------------------

def normalize_landmarks(hand):

    wrist = hand[0]

    relative_points = []
    max_distance = 0

    for landmark in hand:

        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        relative_points.append((x, y, z))

        distance = (x*x + y*y + z*z) ** 0.5

        if distance > max_distance:
            max_distance = distance

    if max_distance == 0:
        max_distance = 1

    normalized = []

    for x, y, z in relative_points:

        normalized.append(x / max_distance)
        normalized.append(y / max_distance)
        normalized.append(z / max_distance)

    return normalized


# --------------------------------
# CAMERA
# --------------------------------

camera = cv2.VideoCapture(0)

previous_timestamp = 0

recording_letter = None
samples_collected = 0
last_sample_time = 0


print()
print("ASL AUTOMATIC DATA COLLECTOR")
print("----------------------------")
print("Make an ASL letter.")
print("Press that letter ONCE.")
print("Then slowly move and tilt your hand.")
print()
print("50 samples will be recorded automatically.")
print("J and Z are skipped for now.")
print("Press ESC to quit.")
print()


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

    hand = None

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        height, width, _ = frame.shape

        # Draw 21 hand landmarks
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


    # --------------------------------
    # AUTOMATIC SAMPLE RECORDING
    # --------------------------------

    if recording_letter is not None:

        cv2.putText(
            frame,
            f"Recording {recording_letter}: "
            f"{samples_collected}/{SAMPLES_PER_LETTER}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        current_time = time.monotonic()

        if (
            hand is not None
            and current_time - last_sample_time >= SAMPLE_INTERVAL
        ):

            landmarks = normalize_landmarks(hand)

            row = [recording_letter] + landmarks

            with open(
                DATA_FILE,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)
                writer.writerow(row)

            samples_collected += 1
            last_sample_time = current_time

            print(
                f"{recording_letter}: "
                f"{samples_collected}/{SAMPLES_PER_LETTER}"
            )

        if samples_collected >= SAMPLES_PER_LETTER:

            print()
            print(
                f"Finished collecting {recording_letter}!"
            )
            print()

            recording_letter = None
            samples_collected = 0


    else:

        cv2.putText(
            frame,
            "Make letter, then press its key",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    cv2.imshow(
        "ASL Data Collector",
        frame
    )

    key = cv2.waitKey(1) & 0xFF


    # ESC quits the program
    if key == 27:
        break


    # --------------------------------
    # START RECORDING A LETTER
    # --------------------------------

    if recording_letter is None:

        if (
            ord("a") <= key <= ord("z")
            or ord("A") <= key <= ord("Z")
        ):

            letter = chr(key).upper()

            if letter in ["J", "Z"]:

                print(
                    f"{letter} uses motion. "
                    "We will add it later."
                )

            else:

                recording_letter = letter
                samples_collected = 0
                last_sample_time = 0

                print()
                print(
                    f"Recording {letter}. "
                    "Slowly move and tilt your hand..."
                )


camera.release()
landmarker.close()
cv2.destroyAllWindows()