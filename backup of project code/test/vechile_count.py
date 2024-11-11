#this code works as single box detection detection with centre representation using Haar Cascade model
import cv2
import numpy as np

# Load the Haar Cascade classifier for car detection
haarcascade_cars = cv2.CascadeClassifier('cars.xml')

# Replace 'your_video.mp4' with the actual path to your video file
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
tracked_vehicles = set()  # To track unique vehicles

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

        # Convert the frame to grayscale for vehicle detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars in the grayscale frame
        cars = haarcascade_cars.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Loop over detected cars
        for (x, y, w, h) in cars:
            vehicle_center = (x + w // 2, y + h // 2)

            # Draw rectangles around detected cars
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame,(vehicle_center),6,(0,0,225),-1)

            #line formula
            line=(point2[1]-point1[1])//(point2[0]-point1[0])

            


            # Check if the vehicle crosses the line
            if (point1[0] <= vehicle_center[0] <= point2[0] or point2[0] <= vehicle_center[0] <= point1[0]):
                if (point1[1] <= vehicle_center[1] <= point2[1] or point2[1] <= vehicle_center[1] <= point1[1]):
                    tracked_vehicles.add((x, y, w, h))  # Track unique vehicles
############################################################################################################################
                    print(tracked_vehicles,"\n")
                    if len(tracked_vehicles) > vehicle_count:
                        vehicle_count += 1

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
