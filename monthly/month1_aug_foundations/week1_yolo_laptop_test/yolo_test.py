from ultralytics import YOLO

# largest object -> often the most relevant for your robot to follow (e.g., the closest person)
# confidence filtering -> avoids reacting to uncertain detections
# center X positioning -> lets you decide movement:
    # if center_x < frame center -> turn left
    # if center_x > frame center -> turn right
    # if near center -> go forward


model = YOLO("yolov8n.pt") # YOLOv8 nano, small and fast for rasp pi
# .pt = pretrained weights, the model learned from COCO dataset (80 object types)
results = model.predict(source=0, imgsz=640, conf=0.8, show=True, stream=True) 
# stream=True means Instead of running all frames at once, YOLO returns them one at a time (so we can process them in a for loop).

for r in results: # Each r is a Results object for one frame of the video feed
    # each r contains:
    # r.boxes => all detected bounding boxes 
    # r.names => mapping from class IDs to labels
    # Other metadata like image size, predictions, masks (if segmentation)

    boxes = r.boxes # a special object containing detection results
    largest_area = 0 # keeps track of the biggest object in this frame
    main_obj = None # will store the label, confidence, and position of the largest object

    # calculating frame center
    width = r.orig_shape[1] # original width of frame
    height = r.orig_shape[0] # original height of frame
    frame_center = width/2
    margin = width * .1
    left_bound = frame_center - margin
    right_bound = frame_center + margin

    for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        # boxes.xyxy: bounding box coordinates [x1, y1, x2, y2]
        # boxes.cls: class ID numbers (e.g., 0 for "person", 2 for "car")
        # boxes.conf: confidence scores for each detection
        # zip() lets us loop over coordinates, class, and confidence together
        area = (box[2] - box[0]) * (box[3] - box[1]) # (x2 - x1) * (y2 - y1)

        # find the largest object, and update the variables
        if area > largest_area:
            largest_area = area
            main_obj = (r.names[int(cls)], float(conf), float((box[0] + box[2]) / 2))

    # if we found at least one detecton print:
        # object label
        # confidence score
        # center X position
    if main_obj:
        label, confidence, center_x = main_obj
        print(f"Object: {label}, Confidence: {confidence:.2f}, Center X: {center_x:.1f}")
