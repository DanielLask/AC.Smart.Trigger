import cv2
import numpy as np
import os
import torch
from multiprocessing import Process
from ultralytics import YOLO
from datetime import datetime  


 #Checks if a bounding box is inside a given ROI
def is_inside_roi(x, y, w, h, roi):
    return x >= roi[0] and y >= roi[1] and w <= roi[2]and h <= roi[3]   



 # Saves a photo with certain annotations if conditions are met.
def photo(Roi, frame, output_dir):
    str_date_time = datetime.now().strftime("%d%m%Y-%H%M%S%f")
    print(f"Object is inside {Roi}")
    name = os.path.join(output_dir, f"{str_date_time}_{Roi}_Trigger.jpg") # File name
    print(name)
    image = np.array(frame)
    cv2.imwrite(name, image)
    
def process_video(source, model, Roi, object_size, minimum_size):
    
    # Draws the rectangle for a specified ROI on the frame. 
    def Region (r):
        cv2.rectangle(annotated_frame,(r[0], r[1]), (r[2], r[3]), (255, 0, 0), 2)
        return annotated_frame
    
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    # Initialize variables here if they are specific to each video stream
    frame_count = 0
    last_detection_time = [None, None]

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()      
    
        if success :
        
            resized_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
            annotated_frame = resized_frame.copy()
         
            frame_count += 1
        
            if frame_count % 2 == 0:
        
                # Run YOLOv8 inference on the frame
                results = model(resized_frame, save=False, show=False, imgsz =320, classes=0, conf=0.7)
                annotated_frame = results[0].plot()
                results = results[0].boxes
                results = results.cpu().numpy()
                # Format cv2.rectangle(frame, start_point (300, 40), end_point (600, 700), color (255, 0, 0), thickness 2)
                Region(Roi[0])
                Region(Roi[1])
            
                     
                if len(results) > 0 :
                
                    for box in results.xyxy.tolist():
                        [x, y, w, h] = box
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        #position = (x, y -10)
                        current_time = datetime.now()
                        
                        #cv2.putText(annotated_frame, text, position, font, font_scale, color, thickness, line_type) 
                        if (w-x)*(h-y) > object_size :
                            #object = resized_frame[y:h, x:w]
                            area= (w-x)*(h-y)
                            print("Calculated area of the object: ",str(area))       
                                                                          
                            if is_inside_roi(x, y, w, h, Roi[0]):
                                if last_detection_time[0] is None or (current_time - last_detection_time[0]).seconds >= delay:
                                    photo("ROI1", annotated_frame, output_dir)
                                    last_detection_time[0] = current_time
                                else:
                                    print("Object detected in ROI1, but skipping due to 10-second rule.")
                            
                            elif is_inside_roi(x, y, w, h, Roi[1]):
                                if last_detection_time[1] is None or (current_time  - last_detection_time[1]).seconds >= delay:
                                    photo("ROI2", annotated_frame, output_dir)
                                    last_detection_time[1] = current_time
                                else:
                                    print("Object detected in ROI2, but skipping due to 10-second rule.")
                            else:
                                print("Object is outside of ROIs")
                        elif (w-x)*(h-y) > minimum_size  :
                        
                            if is_inside_roi(x, y, w, h, Roi[0]) or is_inside_roi(x, y, w, h, Roi[1]) :
                                photo("Training", resized_frame, output_dir2)
                            
                    else:
                        print("No objects detected.")
                else:
                
                    print("No objects detected.")   
        
                    # Display the annotated frame
                cv2.imshow(f"YOLOv8 Inference{source}", annotated_frame)
        
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        
        else:
        # Break the loop if the end of the video is reached
            break


    cap.release()
    cv2.destroyAllWindows()


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    device = torch.device('cpu')
    print('Using device:', device)

# Load the YOLO model
model = YOLO(r"C:\Users\admin\Desktop\last.pt")
# Specify the source (0 for webcam, or path to a video file)

source = [
    r"C:\Users\admin\Desktop\live_Landing_Trigger_L12-10-20230727-130956.mp4",
    r"C:\Users\admin\Desktop\live_Landing_Trigger_L34-10-20230721-142221.mp4",
    r"C:\Users\admin\Desktop\landing.mp4",
    r"rtsp://172.16.30.107/stream0",
    r"rtsp://172.16.30.102:554/stream0"
        ]

object = None
delay = 10
object_size = [
    30000,
    20000
    ]
minimum_size = [
    6000,
    8000
]


# Line 34 Landing ROI
#ROI  format roi1_x1, roi1_y1, roi1_x2, roi1_y2
Roi = [
        [250, 0, 660, 700], # Line2
        [660, 0, 1080, 700] # Line1
]

#Line 34 Landing ROI
#roi1_x1, roi1_y1, roi1_x2, roi1_y2 = 250, 0, 660, 700
#roi2_x1, roi2_y1, roi2_x2, roi2_y2 = 660, 0, 1080, 700

# Set the text, position, font, scale, and color
text = ""
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (255, 255, 255)  # White
thickness = 2
line_type = cv2.LINE_AA



# Specify the directory to save the frames
output_dir = r"C:\Users\Admin\Desktop\frames"
output_dir2 = r"C:\Users\Admin\Desktop\frames\raw"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)

def main():
    p1 = Process(target=process_video, args=(source[3], model, Roi, object_size[0], minimum_size[0]))
    p2 = Process(target=process_video, args=(source[4], model, Roi, object_size[1], minimum_size[0]))
    
    # Start both processes
    p1.start()
    p2.start()

    # Wait for both processes to finish
    p1.join()
    p2.join()
    
if __name__ == "__main__":
    main()