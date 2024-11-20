import cv2
import numpy as np

# Load YOLOv4 model
weights_path = 'yolov4.weights'  # Path to YOLOv4 weights file
config_path = 'yolov4.cfg'       # Path to YOLOv4 config file
net = cv2.dnn.readNet(weights_path, config_path)

# Load the COCO class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Replace with your video file path
video_path = '3727445-hd_1920_1080_30fps.mp4'

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get total frames and FPS
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a resizable window
cv2.namedWindow('Vehicle Detection', cv2.WINDOW_NORMAL)

# Variables for playback
current_frame = 0
speed = 1
paused = False

# Variables for adjustable points
point1 = (100, 100)
point2 = (300, 300)
point_radius = 5
dragging_point1 = False
dragging_point2 = False

# Vehicle counting variables
vehicle_count = 0
tracked_vehicles = {}  # {vehicle_id: (x, y, w, h, crossed)}
vehicle_id_counter = 0

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global point1, point2, dragging_point1, dragging_point2
    if event == cv2.EVENT_LBUTTONDOWN:
        if (x - point1[0]) ** 2 + (y - point1[1]) ** 2 < point_radius ** 2:
            dragging_point1 = True
        elif (x - point2[0]) ** 2 + (y - point2[1]) ** 2 < point_radius ** 2:
            dragging_point2 = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point1:
            point1 = (x, y)
        elif dragging_point2:
            point2 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point1 = False
        dragging_point2 = False

cv2.setMouseCallback('Vehicle Detection', mouse_callback)

# Trackbar callback for seeking
def on_trackbar(val):
    global current_frame
    current_frame = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

cv2.createTrackbar('Seek', 'Vehicle Detection', 0, total_frames - 1, on_trackbar)

def is_crossing_line(center, prev_center):
    """Check if a vehicle crosses the line between point1 and point2."""
    if (min(point1[1], point2[1]) < center[1] < max(point1[1], point2[1]) and
        min(point1[0], point2[0]) < center[0] < max(point1[0], point2[0])):
        if prev_center is not None:
            return (prev_center[1] < min(point1[1], point2[1]) and center[1] >= min(point1[1], point2[1])) or \
                   (prev_center[1] > max(point1[1], point2[1]) and center[1] <= max(point1[1], point2[1]))
    return False

while True:
    if not paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            print("End of video reached.")
            break

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        detections = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        # Collect all detections
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(obj[0] * frame.shape[1])
                    center_y = int(obj[1] * frame.shape[0])
                    w = int(obj[2] * frame.shape[1])
                    h = int(obj[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    label = str(classes[class_id])
                    if label in ['car', 'truck', 'bus', 'motorcycle']:
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        current_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                current_detections.append((x, y, w, h))

        # Update tracked vehicles
        updated_tracked_vehicles = {}
        for (x, y, w, h) in current_detections:
            center = (x + w // 2, y + h // 2)
            matched = False

            for vid, (vx, vy, vw, vh, crossed) in tracked_vehicles.items():
                prev_center = (vx + vw // 2, vy + vh // 2)
                if abs(center[0] - prev_center[0]) < 50 and abs(center[1] - prev_center[1]) < 50:
                    matched = True
                    updated_tracked_vehicles[vid] = (x, y, w, h, crossed)

                    # Check for crossing
                    if not crossed and is_crossing_line(center, prev_center):
                        updated_tracked_vehicles[vid] = (x, y, w, h, True)
                        vehicle_count += 1
                    break

            if not matched:
                vehicle_id_counter += 1
                updated_tracked_vehicles[vehicle_id_counter] = (x, y, w, h, False)

        tracked_vehicles = updated_tracked_vehicles

        # Draw bounding boxes
        for vid, (x, y, w, h, crossed) in tracked_vehicles.items():
            color = (0, 255, 0) if crossed else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID {vid}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw the adjustable line
        cv2.circle(frame, point1, point_radius, (0, 0, 255), -1)
        cv2.circle(frame, point2, point_radius, (255, 0, 0), -1)
        cv2.line(frame, point1, point2, (0, 255, 255), 2)

        # Display vehicle count
        cv2.putText(frame, f"Vehicles Crossed: {vehicle_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        # Playback speed
        cv2.putText(frame, f"Speed: {speed}x", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.setTrackbarPos('Seek', 'Vehicle Detection', current_frame)

        cv2.imshow('Vehicle Detection', frame)
        current_frame += speed
        if current_frame >= total_frames:
            current_frame = total_frames - 1

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
    elif key == ord('+'):
        speed = min(speed + 1, 5)
    elif key == ord('-'):
        speed = max(speed - 1, 1)
    elif key == ord('r'):
        current_frame = 0
    elif key == ord('c'):
        vehicle_count = 0
        tracked_vehicles.clear()

cap.release()
cv2.destroyAllWindows()
