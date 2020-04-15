import os
import cv2
import fnmatch
import numpy as np
import imutils


def drawBoundingBoxes(labels, stats, frame):
	for l in labels:
		x = stats[l, cv2.CC_STAT_LEFT]
		y = stats[l, cv2.CC_STAT_TOP]
		w = stats[l, cv2.CC_STAT_WIDTH]
		h = stats[l, cv2.CC_STAT_HEIGHT]
		cv2.putText(frame,'%s'%('CheckObject'), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2 )
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

def findConnectedComponents(frame, components_area_lower, components_area_upper):
	connected_components = cv2.connectedComponentsWithStats(frame, 4, cv2.CV_8S) #List of Connected Components background
	num_connected_components = connected_components[0]
	connected_components_labels = list(range(1, num_connected_components)) #--> List of components labels
	connected_components_stats = connected_components[2]			# Stats of labels
	connected_components_labels = list(filter(lambda x: connected_components_stats[x, cv2.CC_STAT_AREA] > components_area_lower and connected_components_stats[x, cv2.CC_STAT_AREA] < components_area_upper , connected_components_labels))
	
	return connected_components_labels, connected_components_stats, num_connected_components




def main():
     
	bg_1 = '../data/bg/ABODA/video7/frm_00681.jpg'
	bg_2 = '../data/bg/ABODA/video7/frm_04004.jpg'
	frame_t = '../data/frames/video7/frm_03206.jpg'
	

	

	while True:
          bg1 = cv2.imread(bg_1)
          bg2 = cv2.imread(bg_2)          
          frame = cv2.imread(frame_t)
          
          


          canny1 = cv2.Canny(bg1, 10, 200)
          canny2 = cv2.Canny(bg2, 10, 200)
          diff_canny = cv2.absdiff(canny2, canny1)
          
          



         


          cv2.imshow('frame', diff_canny) 
          #cv2.imshow('diff', diff_canny) 
          k = cv2.waitKey(1)
          if k == ord('q'):
              break

          
	  

	  
		
		 
  
		
      


if __name__ == '__main__':
	main()