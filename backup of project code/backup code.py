#this code works as multiple box detection without centre representation using haar YOLOv4 model

import cv2
import numpy as np

# Load YOLOv4 model
weights_path = 'yolov4.weights'  # Path to the YOLOv4 weights file
config_path = 'yolov4.cfg'        # Path to the YOLOv4 configuration file
net = cv2.dnn.readNet(weights_path, config_path)

# Load the COCO names file for class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Replace 'video.mp4' with your video file path
video_path = '3727445-hd_1920_1080_30fps.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get total frames and FPS
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a resizable window
cv2.namedWindow('Vehicle Detection', cv2.WINDOW_NORMAL)

# Variables to manage playback
current_frame = 0
speed = 1  # Playback speed factor
paused = False

# Variables for adjustable points
point1 = (100, 100)  # Initial coordinates for point 1
point2 = (300, 300)  # Initial coordinates for point 2
point_radius = 5  # Radius for points
dragging_point1 = False
dragging_point2 = False

# Variables for vehicle counting
vehicle_count = 0
crossed_vehicles = set()  # To track vehicles that have crossed

# Define a dictionary to keep track of vehicle positions
tracked_vehicles = {}

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global point1, point2, dragging_point1, dragging_point2

    # Check if a point is being dragged
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the mouse is near point1
        if (x - point1[0]) ** 2 + (y - point1[1]) ** 2 < point_radius ** 2:
            dragging_point1 = True
        # Check if the mouse is near point2
        elif (x - point2[0]) ** 2 + (y - point2[1]) ** 2 < point_radius ** 2:
            dragging_point2 = True

    elif event == cv2.EVENT_MOUSEMOVE:
        # Update point1 if it is being dragged
        if dragging_point1:
            point1 = (x, y)
        # Update point2 if it is being dragged
        elif dragging_point2:
            point2 = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # Stop dragging when the mouse button is released
        dragging_point1 = False
        dragging_point2 = False

# Set the mouse callback for the window
cv2.setMouseCallback('Vehicle Detection', mouse_callback)

# Trackbar callback to update current frame
def on_trackbar(val):
    global current_frame
    current_frame = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

# Create a trackbar for video seeking
cv2.createTrackbar('Seek', 'Vehicle Detection', 0, total_frames - 1, on_trackbar)

while True:
    if not paused:
        # Set the current frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            print("End of video reached.")
            break

        # Prepare the frame for YOLOv4
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Get the output layers
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

        # Run the model
        detections = net.forward(output_layers)

        # Loop over detections and draw bounding boxes for vehicles
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Confidence threshold
                    center_x = int(obj[0] * frame.shape[1])
                    center_y = int(obj[1] * frame.shape[0])
                    w = int(obj[2] * frame.shape[1])
                    h = int(obj[3] * frame.shape[0])

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    label = str(classes[class_id])
                    if label in ['car', 'truck', 'bus', 'motorcycle']:  # Filter vehicle classes
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Track vehicle ID using unique identifier (class_id + center_y)
                        vehicle_id = (class_id, center_y)
                        
                        # Determine if the vehicle crosses the line
                        vehicle_center = (center_x, center_y)
                        if (point1[0] <= vehicle_center[0] <= point2[0] or point2[0] <= vehicle_center[0] <= point1[0]):
                            if (point1[1] <= vehicle_center[1] <= point2[1] or point2[1] <= vehicle_center[1] <= point1[1]):
                                # Check if the vehicle is already counted
                                if vehicle_id not in tracked_vehicles:
                                    vehicle_count += 1
                                    tracked_vehicles[vehicle_id] = True

        # Draw adjustable points
        cv2.circle(frame, point1, point_radius, (0, 0, 255), -1)  # Point 1 (Red)
        cv2.circle(frame, point2, point_radius, (255, 0, 0), -1)  # Point 2 (Blue)

        # Draw a line connecting the two points
        cv2.line(frame, point1, point2, (0, 255, 255), 2)  # Line (Yellow)

        # Display vehicle count
        cv2.putText(frame, f"Vehicles Crossed: {vehicle_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the current speed on the video
        text = f"Speed: {speed}x"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Update the trackbar position
        cv2.setTrackbarPos('Seek', 'Vehicle Detection', current_frame)

        # Show the video frame with detections
        cv2.imshow('Vehicle Detection', frame)

        # Update current frame based on speed
        current_frame += speed

        # Ensure current_frame does not exceed total_frames
        if current_frame >= total_frames:
            current_frame = total_frames - 1

    # Handle key events
    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):  # Quit
        break
    elif key == ord('p'):  # Play/Pause
        paused = not paused
    elif key == ord('+'):  # Increase speed
        speed = min(speed + 1, 5)  # Limit max speed to 5x
    elif key == ord('-'):  # Decrease speed
        speed = max(speed - 1, 1)  # Limit min speed to 1x
    elif key == ord('r'):  # Restart video
        current_frame = 0
    elif key == ord('c'):  # Reset vehicle count
        vehicle_count = 0
        tracked_vehicles.clear()  # Clear the tracked vehicles

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()


# initially in first frame the vehicle is detected and box is drawn
# in next frame again the same vehicle is detected and one more box is drawn
# make sure that the previous detection is carried forward to next frame
# and once detected vehicle is not detected as the different vehicle
