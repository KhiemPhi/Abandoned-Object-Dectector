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
     
	bg_1 = '../data/bg/video7/frm_00003.jpg'
	bg_2 = '../data/bg/video7/frm_03206.jpg'
	frame_t = '../data/frames/video7/frm_03206.jpg'
	

	

	while True:
          bg1 = cv2.imread(bg_1)
          bg2 = cv2.imread(bg_2)          
          frame = cv2.imread(frame_t)
          bg1 = cv2.cvtColor(bg1, cv2.COLOR_RGB2GRAY)
          bg2 = cv2.cvtColor(bg2, cv2.COLOR_RGB2GRAY)
          diff = cv2.absdiff(bg1, bg2)
          diff_canny = cv2.Canny(diff, 10, 100)
          #diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
          labels, stats, num = findConnectedComponents(diff_canny, 10, 200)
          bg1 = cv2.cvtColor(bg1, cv2.COLOR_GRAY2RGB)
          bg2 = cv2.cvtColor(bg2, cv2.COLOR_GRAY2RGB)
          diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)	
          drawBoundingBoxes(labels,stats, frame)
          diff_canny = cv2.cvtColor(diff_canny, cv2.COLOR_GRAY2RGB)
          stack = np.hstack((frame, diff_canny, diff, bg1, bg2))


          cv2.imshow('frame', stack) 
          #cv2.imshow('diff', diff_canny) 
          k = cv2.waitKey(1)
          if k == ord('q'):
              break

          
	  

	  
		
		 
  
		
      


if __name__ == '__main__':
	main()