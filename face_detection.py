import cv2

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Function to process video and detect people
def detect_people(video_source):
    # Open the video source (0 for webcam, or file path for video)
    cap = cv2.VideoCapture(video_source)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 15  # Default to 15 if FPS not available
    
    # Define the codec and create VideoWriter object to save output
    # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    
    while cap.isOpened():
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        
        # Resize frame for faster detection (to 640x480)
        scale_factor = frame_width / 640
        resized_frame = cv2.resize(frame, (640, 480))
        
        # Detect people in the frame
        boxes, weights = hog.detectMultiScale(resized_frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        
        # Process each detected person
        for (x, y, w, h) in boxes:
            # Scale coordinates back to original frame size
            x = int(x * scale_factor)
            y = int(y * scale_factor)
            w = int(w * scale_factor)
            h = int(h * scale_factor)
            
            # Calculate the top 30% of the height (likely containing the head)
            head_height = int(h * 0.3)
            
            # Define the region of interest (ROI) for the head
            head_y_start = y
            head_y_end = y + head_height
            head_x_start = x
            head_x_end = x + w
            
            # Ensure ROI stays within frame bounds
            head_y_start = max(0, head_y_start)
            head_y_end = min(frame_height, head_y_end)
            head_x_start = max(0, head_x_start)
            head_x_end = min(frame_width, head_x_end)
            
            # Extract the head region
            head_region = frame[head_y_start:head_y_end, head_x_start:head_x_end]
            
            # Apply Gaussian blur to the head region
            if head_region.size > 0:  # Check if region is valid
                blurred_head = cv2.GaussianBlur(head_region, (51, 51), 0)
                frame[head_y_start:head_y_end, head_x_start:head_x_end] = blurred_head
            
            # Draw a rectangle around the head region
            cv2.rectangle(frame, (head_x_start, head_y_start), (head_x_end, head_y_end), (0, 255, 0), 2)
        
        # Write the frame to the output video
        # out.write(frame)
        
        # Display the frame
        # cv2.imshow('People Detection (Head Blurred)', frame)
        
        # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Release everything when done

    cap.release()
    # out.release()
    cv2.destroyAllWindows()


    
# Function to process video and detect people
def blur_frame(frame, frame_width, frame_height):
    # Open the video source (0 for webcam, or file path for video)
    
    # # Check if video opened successfully
    # if not cap.isOpened():
    #     print("Error: Could not open video source.")
    #     return
    
    # # Get video properties for output
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 15  # Default to 15 if FPS not available
    
    # Define the codec and create VideoWriter object to save output
    # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    
    # Resize frame for faster detection (to 640x480)
    scale_factor = frame_width / 640
    resized_frame = cv2.resize(frame, (640, 480))
    
    # Detect people in the frame
    boxes, weights = hog.detectMultiScale(resized_frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    
    # Process each detected person
    for (x, y, w, h) in boxes:
        # Scale coordinates back to original frame size
        x = int(x * scale_factor)
        y = int(y * scale_factor)
        w = int(w * scale_factor)
        h = int(h * scale_factor)
        
        # Calculate the top 30% of the height (likely containing the head)
        head_height = int(h * 0.3)
        
        # Define the region of interest (ROI) for the head
        head_y_start = y
        head_y_end = y + head_height
        head_x_start = x
        head_x_end = x + w
        
        # Ensure ROI stays within frame bounds
        head_y_start = max(0, head_y_start)
        head_y_end = min(frame_height, head_y_end)
        head_x_start = max(0, head_x_start)
        head_x_end = min(frame_width, head_x_end)
        
        # Extract the head region
        head_region = frame[head_y_start:head_y_end, head_x_start:head_x_end]
        
        # Apply Gaussian blur to the head region
        if head_region.size > 0:  # Check if region is valid
            blurred_head = cv2.GaussianBlur(head_region, (51, 51), 0)
            frame[head_y_start:head_y_end, head_x_start:head_x_end] = blurred_head
        
        # Draw a rectangle around the head region
        cv2.rectangle(frame, (head_x_start, head_y_start), (head_x_end, head_y_end), (0, 255, 0), 2)

    return frame