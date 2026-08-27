import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import serial

from collections import deque, Counter

DATA_FILE = "asl_landmarks.csv"

# --------------------------------
# ARDUINO CONNECTION
# --------------------------------

arduino = serial.Serial("COM3", 115200)
time.sleep(2)

# --------------------------------
# LOAD TRAINING DATA
# --------------------------------

X = []
y = []

with open(DATA_FILE, "r") as file:
    reader = csv.reader(file)

    for row in reader:
        if not row:
            continue

        label = row[0]
        landmarks = [float(value) for value in row[1:]]

        X.append(landmarks)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Samples loaded:", len(X))
print("Letters available:", sorted(set(y)))

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

    return np.array(
        normalized,
        dtype=np.float32
    )

# --------------------------------
# KNN CLASSIFIER
# --------------------------------

def predict_letter(sample, k=3):

    distances = np.linalg.norm(
        X - sample,
        axis=1
    )

    nearest_indices = np.argsort(
        distances
    )[:k]

    nearest_labels = y[nearest_indices]

    labels, counts = np.unique(
        nearest_labels,
        return_counts=True
    )

    return labels[np.argmax(counts)]

# --------------------------------
# MEDIAPIPE
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
# CAMERA
# --------------------------------

camera = cv2.VideoCapture(0)

previous_timestamp = 0

# --------------------------------
# TEMPORAL SMOOTHING
# --------------------------------

prediction_history = deque(maxlen=7)

stable_letter = None
last_sent_letter = None

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

    timestamp = int(
        time.monotonic() * 1000
    )

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

        sample = normalize_landmarks(hand)

        raw_prediction = predict_letter(sample)

        prediction_history.append(raw_prediction)

        if len(prediction_history) == 7:

            counts = Counter(prediction_history)

            most_common_letter, count = (
                counts.most_common(1)[0]
            )

            if count >= 6:
                stable_letter = most_common_letter

        # --------------------------------
        # SEND STABLE LETTER TO ARDUINO
        # --------------------------------

        if (
            stable_letter is not None
            and stable_letter != last_sent_letter
        ):

            arduino.write(
                stable_letter.encode("utf-8")
            )

            print(
                "Sent to Arduino:",
                stable_letter
            )

            last_sent_letter = stable_letter

        # --------------------------------
        # DISPLAY
        # --------------------------------

        cv2.putText(
            frame,
            f"Raw: {raw_prediction}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        if stable_letter is not None:

            cv2.putText(
                frame,
                f"ASL: {stable_letter}",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3
            )

        for landmark in hand:

            x = int(landmark.x * width)
            y_pos = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y_pos),
                5,
                (0, 255, 0),
                -1
            )

    else:

        prediction_history.clear()
        stable_letter = None
        last_sent_letter = None

    cv2.imshow(
        "ASL Recognizer",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
arduino.close()
landmarker.close()
cv2.destroyAllWindows()