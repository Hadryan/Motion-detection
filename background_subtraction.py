import cv2
import numpy as np

# Define a function to get the current frame from the webcam
def get_frame(cap, scaling_factor):
    # Read the current frame from the video capture object
    _, frame = cap.read()

    # Resize image
    frame = cv2.resize(frame, None, fx=scaling_factor, 
            fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return frame

if __name__=='__main__':
  
    cap = cv2.VideoCapture(0)

  
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
     
    
    history = 100

  
    learning_rate = 1.0/history

    
    while True:
        # Grab the current frame
        frame = get_frame(cap, 0.5)

        # Compute the mask 
        mask = bg_subtractor.apply(frame, learningRate=learning_rate)

        # Convert grayscale image to RGB color image
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Display image
        cv2.imshow('Input', frame)
        cv2.imshow('Output', mask & frame)

 
        c = cv2.waitKey(10)
        if c == 27:
            break

  
    cap.release()
    
   
    cv2.destroyAllWindows()
