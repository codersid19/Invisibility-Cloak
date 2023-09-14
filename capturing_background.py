import cv2
import time
import os

# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 represents the default camera, change it if necessary

# Set start time
start_time = time.time()

# Create a folder to save frames
save_folder = 'foreground'
os.makedirs(save_folder, exist_ok=True)

# Capture frames until 30 seconds have passed
frame_count = 0
while (time.time() - start_time) < 30:
    # Read a frame from the camera
    ret, frame = camera.read()

    # Save the frame in the folder
    frame_path = os.path.join(save_folder, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_path, frame)

    # Increment the frame count
    frame_count += 1

# Release the camera
camera.release()

# Print the total number of frames captured
print(f"Total frames captured: {frame_count}")

