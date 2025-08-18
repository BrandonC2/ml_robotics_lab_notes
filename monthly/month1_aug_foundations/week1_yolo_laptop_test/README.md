
# created python venv and activated it
python -m venv mlrobotics
source mlrobotics/bin/activate

# what to install
pip install ultralytics opencv-python

# to check if installed properly
python -c "import cv2; import ultralytics; print('OpenCV & YOLO ready')"

# notes:

for every bounding box candidate, YOLO outputs:
[class probabilities] + [box coordinates]


## in detection you balance:
* precision = how many of the detections were correct
* recall = of actual objects in the image, how many did you find

trade-off:
* high conf => fewer false positives
* low conf => catch more objects but risk chasing ghosts

why it matters:
* too low conf -> robot reacts to shadows or random textures
* too high -> robot freezes because it "doesnt see" its target enough
* middle ground (0.4 - 0.6) often works best in real environments

practical advice:
* start high (conf=0.7) during development to be sure your detections are correct
* lower gradually as you want the robot to react to more cases
* for moving robots: combine conf with temporal smoothing - 
    e.g. require the target to be seen for N frames before acting

ex output for one box:
[0.05, 0.12, 0.73, 0.05, 0.05] + [x1, y1, x2, y2]
* Each number in the first part is probability of a class (e.g., dog, person, car…)
* The highest probability is 0.73 → meaning YOLO is 73% sure this box is that class.

x1,y1 => top-left corner of the box
x2,y2 => bottom-right corner of the box

for example: 
if a YOLO detects a cup and gives:
[100, 50, 180, 130]

left edge = 100 pixels
top edge = 50 pixels
right edge = 180 pixels
bottom edge = 130 pixels

width = x2 - x1 = 180 - 100 = 80 pixels
height =  y2 - y1 = 130 - 50 = 80 pixels

area = width * height
=> area = (box[2] - box[0]) * (box[3] - box[1])

example #2
If YOLO detects a bottle:
[100, 50, 180, 200]

left edge is at 100px from the left of the frame
right edge is at 180 px from the left of the frame

## what is center_x
center_x is the horizontal middle of the object in the frame

formula:
center_x = (x1 + x2) / 2

for example (from prev example):
center_x = (100 + 180) / 2
=> 140 pixels

the center of the bottle is at 140 px from the left edge of the frame 

center_x < frame_center → Object is to the left → turn left
center_x > frame_center → Object is to the right → turn right
center_x ≈ frame_center → Object is in front → go forward

if your frame is 640 px wide,
frame_center = 640 / 2 = 320 px

then, 
center_x = 140 → turn left
center_x = 500 → turn right
center_x = 320 → go forward

## results = model.predict(source=0, imgsz=640, conf=0.5, show=True, save=True, project="week1_tests")

source=0  => "open webcam at index 0" this is the main camera

show=True => opens a window to display each frame with YOLO's boxes, labels, and confidence scores

imgsz=640 => (default 640x640) smaller imgsz for faster, but less accurate detections
* imgsz resizes + pad (if not perfect square) to feed the neural net
* the network predicts bounding boxes in this resized space
* then YOLO maps the boxes back to your original frame size before returning them
* results[0].boxes.xyxy are in the original frame coordinates
* after prediction, YOLO rescales everything to match camera's original resolution


conf=.x => higher to filter out weak detections
* ex: conf=.5, filters out anything less than .5 confidence score

save=True, project="folder name" => saves a mp4 video in a folder you named

## object detection pipeline:
Webcam → YOLOv8 Model → Detections (boxes, labels, confidence) → Output
                                                    ↓
                                Display live window + Save to folder

Inference: Passing your webcam frames through YOLO’s layers to get:
* Bounding boxes: where objects are
* Labels: what they are
* Confidence: how sure the model is

## what's happening:
YOLOv8n takes each video frame and:
1. resizes it to the chosen imgsz
2. normalizes pixel values to between 0-1
3. runs through convolutional layers => extract patterns (edges, colors, shapes)
4. applies detection heads => predicts bounding boxes, classes, and confidences
5. runs non-max suppression (NMS) => removes overlapping duplicate boxes
6. returns result as a structured python object

## why save results?
we save results to:
* replay situations for debugging
* see false positives/negatives
* create custom datasets by labeling these frames and retraining YOLO for specific tasks

## tolerance band
tolerance band is a margin that stops the robot from overreacting.

for example, if object's center_x is:
321 -> TURN RIGHT
319 -> TURN LEFT

it would make the robot jittery

## how big should margin be?
depends on your frame width (resolution):

for 640 px wide frame: margin ≈ 40–60 px

for 1280 px wide frame: margin ≈ 80–120 px

rule of thumb:
* Margin ≈ 10% of frame width