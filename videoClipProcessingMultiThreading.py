import cv2
import time
import threading
from ultralytics import YOLO

#Load Model
model = YOLO('yolov8n-pose.pt')

# Open the camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened(): 
    print("Unable to read camera feed")

# Default resolutions of the frame are obtained (system dependent)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter objects
fourcc = cv2.VideoWriter_fourcc(*'XVID')

clip_number = 0
start_time = time.time()

# Define 'out' before the while loop
out = cv2.VideoWriter('clip_{}.avi'.format(clip_number), fourcc, 20.0, (frame_width, frame_height))

#create a list to store the clips
video_buffer=[]

def process_clip(clip_number):
    results = model.track('clip_{}.avi'.format(clip_number-1),save=True)
    for r in results:
        print(r.names)

while True:
    ret, frame = cap.read()
    if ret:
        # Write the frame into the file 'output.avi'
        out.write(frame)

        # Display the resulting frame    
        cv2.imshow('frame', frame)

        # If it's been 1 second since the last clip started, start a new clip
        if time.time() - start_time >= 1:

            clip_number += 1
            out = cv2.VideoWriter('clip_{}.avi'.format(clip_number), fourcc, 20.0, (frame_width, frame_height))
            
            # Start a new thread to process the previous clip
            threading.Thread(target=process_clip, args=(clip_number,)).start()

            start_time = time.time()

        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break  

# Release everything when job is finished
out.release()
cap.release()
cv2.destroyAllWindows() 
