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

  
    scaling_factor = 0.5

  
    while True:
        # Grab the current frame
        frame = get_frame(cap, scaling_factor)

        # Convert the image to HSV colorspace
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of skin color in HSV
        lower = np.array([0, 70, 60])
        upper = np.array([50, 150, 255])

        # Threshold the HSV image to get only skin color
        mask = cv2.inRange(hsv, lower, upper)

        # Bitwise-AND between the mask and original image
        img_bitwise_and = cv2.bitwise_and(frame, frame, mask=mask)

        # Run median blurring
        img_median_blurred = cv2.medianBlur(img_bitwise_and, 5)

      
        cv2.imshow('Input', frame)
        cv2.imshow('Output', img_median_blurred)

       
        c = cv2.waitKey(5)
        if c == 27:
            break

   
    cv2.destroyAllWindows()
