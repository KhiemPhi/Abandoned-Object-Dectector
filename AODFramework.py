import cv2
import numpy as np
import os
from skimage.measure import compare_ssim

cap = cv2.VideoCapture('Test5.avi')
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter("outputnew.avi", fourcc, 5.0, (1280,720))
path = 'C:/Khiem/Illegal Dumping/Frames3Background'

cv2.namedWindow('Abandoned Object Detection',cv2.WINDOW_NORMAL)
cv2.namedWindow('Background Subtracted Model', cv2.WINDOW_NORMAL)
_, first_frame = cap.read()
#cv2.imwrite("Test5Frame0.jpg", first_frame)
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

fgbg = cv2.createBackgroundSubtractorKNN()
fgbg.setDetectShadows(False)


numberOfContours = 0
suspected_frames = []
nextFrameToCheck = []
frameWindow = 100
frameNumber = 0
suspected_items = []

def variance (rect1, rect2):
    (x1, y1, w1, h1) = rect1
    (x2, y2, w2, h2) = rect2
    scoreX = (x1-x2)**2
    scoreY = (y1-y2)**2
    scoreW = (w1-w2)**2
    scoreH = (h1-h2)**2
    total = scoreX + scoreY + scoreW + scoreH
    return total

def check_variance(rect1, listRect):
    similar_rect = []
    for rect in listRect:
        if (variance(rect1, rect) < 600):
           return True
    return False

while cap.isOpened():
    ret, frame = cap.read()    

    # Keeping track of the Frame Number
    frameNumber = frameNumber + 1 

    # Condition To Stop Reading Capture
    if frame is None:
        break        
       
    #=============================================================
    # (1) Storing Contours + Creating Flag:
    #       - (a) Find all the contours in the background model that has been dilated and eroded
    #       - (b) Filter out all contours that are too large or too small
    #       - (c) Check if number of filtered contours is less than the previous frame's number of filtered contours, if it is less 
    #       then mark the frame as suspected and check back again 10 frames later.  
    #==============================================================
    
    img_diff = cv2.absdiff(frame, first_frame)    
    img_diff_edges = cv2.Canny(img_diff, 10, 200)
    dilated = cv2.dilate(img_diff_edges, None, iterations=2)

    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filter_contours = list(filter(lambda x: cv2.contourArea(x) > 10 and cv2.contourArea(x) < 20000, contours)) # Filter out contours that are too large and too small
    

    #If there has been more differences, then check 10 frames from now
    if(numberOfContours >= len(filter_contours) and frameNumber > 0): #If previous contours are less than contours before, then make the flag
        #cv2.putText(frame, "Status: {}".format('Suspect'), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)        
        #Store This Suspected Frame
        suspected_frames.append( frame.copy() ) # Store this suspected frame, check back at 10 frames later
        nextFrameToCheck.append( frameNumber + frameWindow ) # Mark the next frame to be checked   
    

    #=============================================================
    # (2) Checking To See If Contours Are Still There:
    #       - (a) After 10 frames, compared the store selected frames with the current frame
    #       - (b) Check for previous contours are still at the same position
    #       - (c) These contours are abandoned, and now must be 
    #       stored in a list of abandoned contours. 
    #==============================================================  
    if frameNumber in nextFrameToCheck:
        
        previousFrame = suspected_frames.pop(0)
        nextFrameToCheck.pop(0) # Remove this element from the list, don't need to check this frame again
        
        img_diff_past = cv2.absdiff(first_frame, previousFrame)         
        img_diff_edges_past = cv2.Canny(img_diff_past, 200, 500)
        dilated_past = cv2.dilate(img_diff_edges_past, None, iterations=2)   
        contours_past, _ = cv2.findContours(dilated_past.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        filter_contours_past = list(filter(lambda x: cv2.contourArea(x) > 10 and cv2.contourArea(x) < 20000, contours_past)) # Filter out contours that are too large and too small
        
        #Check Same Contours:
        rectangles_past = list(map(lambda x: cv2.boundingRect(x), filter_contours_past))
        rectangles_current = list(map(lambda x: cv2.boundingRect(x), filter_contours))
        same_rectangles = list(filter(lambda x: check_variance(x, rectangles_past), rectangles_current))
        

        for suspected_item in same_rectangles:
            (x, y, w, h) = suspected_item
            cv2.putText(frame,'%s'%('CheckObject'), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3) 

        
        
        '''        
        file_name = "detection_" + str("{:05d}".format(frameNumber)) + ".jpg"
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        file_save = np.hstack((frame_gray, dilated))
        file_save = np.hstack((file_save, dilated_recent))                    
        cv2.imwrite(os.path.join(path , file_name), file_save)
        '''
                
          
    # Updating the number of contours 
    numberOfContours = len(filter_contours)  
    cv2.imshow("Abandoned Object Detection", frame)    
    cv2.imshow("Difference Model", img_diff_edges)
   
    


    if cv2.waitKey(1) == ord('q'):
        break
    

cv2.destroyAllWindows()
cap.release()
out.release()