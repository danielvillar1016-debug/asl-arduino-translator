# ASL Arduino Translator

Real-time ASL fingerspelling recognition using MediaPipe, Python, and Arduino.

The system tracks 21 hand landmarks from a laptop camera, classifies static ASL letters, stabilizes predictions across frames, and sends the recognized letter to an Arduino-connected 16x2 LCD.

## Demo



https://github.com/user-attachments/assets/ffb670f5-8b3b-4139-9f4e-ad132a210fb3



## Current Support

24 static ASL letters  
Real-time hand tracking  
Custom training dataset  
KNN-based classification  
Temporal smoothing  
Arduino LCD output

J and Z are not yet supported because they require motion tracking across multiple frames.

## Materials

### Required

Arduino board  
USB cable  
16x2 LCD  
Jumper wires  
Resistor for LCD backlight  
Laptop with camera

### Optional

Breadboard  
External webcam  
Potentiometer for LCD contrast

## Software

Python 3  
Arduino IDE  
VS Code

Install required Python packages:

```bash
python -m pip install opencv-python mediapipe numpy pyserial
```

## Project Files

camera_test.py — tests webcam access  
hand_tracking.py — MediaPipe hand tracking prototype  
collect_data.py — collects labeled ASL landmark samples  
recognize_asl.py — runs live ASL recognition and sends output to Arduino  
asl_landmarks.csv — collected training data  
hand_landmarker.task — MediaPipe hand landmark model  
requirements.txt — Python dependencies  
arduino/ — Arduino LCD sketch

## Data Collection

Run:

```bash
python collect_data.py
```

Make an ASL handshape.  
Press the corresponding keyboard letter once.  
Slowly vary the hand angle and distance.  
The program records 50 samples automatically.  
Repeat for each static letter.

J and Z are skipped.

## Running the Project

Upload the Arduino LCD sketch.

Close Arduino Serial Monitor.

Then run:

```bash
python recognize_asl.py
```

The recognized letter appears in the camera window and on the Arduino LCD.

## LCD Wiring

Arduino LCD configuration:

```cpp
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);
```

| LCD Pin | Connection |
|---|---|
| VSS | GND |
| VDD | 5V |
| VO | GND |
| RS | Arduino 12 |
| RW | GND |
| E | Arduino 11 |
| D4 | Arduino 5 |
| D5 | Arduino 4 |
| D6 | Arduino 3 |
| D7 | Arduino 2 |
| A | 5V through resistor |
| K | GND |

D0-D3 are unused.

## Recognition Method

MediaPipe tracks 21 hand landmarks with x, y, and z coordinates.

Landmarks are normalized relative to the wrist and hand size.

A K-nearest-neighbors classifier compares the live hand landmarks against the collected training samples.

A prediction is accepted when the same letter appears in at least 6 of the previous 7 frames.

## Limitations

J and Z require motion recognition  
Training data is currently based primarily on one user's hand  
Extreme camera angles can reduce accuracy

## Future Improvements

J and Z motion recognition  
Word construction  
Multi-user training data  
Confidence thresholding
