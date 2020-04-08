import os
import cv2
import fnmatch
import numpy as np
import imutils
from skimage.measure import compare_ssim

def drawBoundingBoxes(labels, stats, frame):
	for l in labels:
		x = stats[l, cv2.CC_STAT_LEFT]
		y = stats[l, cv2.CC_STAT_TOP]
		w = stats[l, cv2.CC_STAT_WIDTH]
		h = stats[l, cv2.CC_STAT_HEIGHT]
		cv2.putText(frame,'%s'%('CheckObject'), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2 )
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

def drawBoundingBoxes1(labels, stats, frame):
	for l in labels:
		x = stats[l, cv2.CC_STAT_LEFT]
		y = stats[l, cv2.CC_STAT_TOP]
		w = stats[l, cv2.CC_STAT_WIDTH]
		h = stats[l, cv2.CC_STAT_HEIGHT]
		cv2.putText(frame,'%s'%('CheckObject'), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2 )
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

def findConnectedComponents(frame, components_area_lower, components_area_upper):
	connected_components = cv2.connectedComponentsWithStats(frame, 4, cv2.CV_8S) #List of Connected Components background
	num_connected_components = connected_components[0]
	connected_components_labels = list(range(1, num_connected_components)) #--> List of components labels
	connected_components_stats = connected_components[2]			# Stats of labels
	connected_components_labels = list(filter(lambda x: connected_components_stats[x, cv2.CC_STAT_AREA] > components_area_lower and connected_components_stats[x, cv2.CC_STAT_AREA] < components_area_upper , connected_components_labels))
	return connected_components_labels, connected_components_stats, num_connected_components

def main():
	frm_dir = '../data/frames/video1/'
	sub_dir = '../data/subtract/video1/'

	frm_files = sorted(fnmatch.filter(os.listdir(frm_dir), '*.jpg'))

	frame_0 = cv2.imread(os.path.join(frm_dir, frm_files[0]))
	
	fgbg = cv2.createBackgroundSubtractorMOG2()
	# fgbg2 = cv2.createBackgroundSubtractorMOG2()

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	# fgmask = fgbg.apply(frame_0)

	total_bg_connected_components_past = 0	
	total_fg_connected_components_past = 0
	frame_number = 0
	check_window = 10
	frames_to_check = dict()
	diff = np.zeros(frame_0.shape, np.uint8)
	components_area_lower = 100
	components_area_upper = 200

	for frm_file in frm_files[1:]:
		# fgmask = fgbg.apply(frame_0)
		frame_t = cv2.imread(os.path.join(frm_dir, frm_file))
		bkg = fgbg.getBackgroundImage() #---> Get Background Image
		bkg_Canny = cv2.Canny(bkg, 10, 200) #---> Get Background Edges	
		
		fgmask = fgbg.apply(frame_t) # Find the foreground mask

		#Thresholding : Foreground Mask
		fgmask[fgmask < 128] = 0
		fgmask[fgmask != 0] = 255

		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
		fgmask_rgb = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2RGB) # Convert to same channel to Stack with frame + bkg image

		# Get Forground Connected Components
		connected_components_labels_fg, connected_components_stats_fg, num_connected_components_fg = findConnectedComponents(fgmask, components_area_lower, components_area_upper)


		# Get Background Connected Components
		if bkg is not None:			

			connected_components_labels_bg, connected_components_stats_bg, num_connected_components_bg = findConnectedComponents(bkg_Canny, components_area_lower, components_area_upper)		
			#drawBoundingBoxes(connected_components_labels_bg, connected_components_stats_bg, frame_t)
			
			# If current bg components is greater than the past bg components then we add the current frame as a suspected frame to our dict
			if (total_bg_connected_components_past < len(connected_components_labels_bg) ):
				total_fg_connected_components_past = len(connected_components_labels_fg)
				total_bg_connected_components_past = len(connected_components_labels_bg)
				suspected_frame = frame_number + check_window
				frames_to_check.update({suspected_frame: (connected_components_labels_bg, connected_components_stats_bg, bkg_Canny) })

			if (frame_number in frames_to_check): # If the frame we are in is a suspected frame, we check the no. of connected components
				connected_components_labels_check, connected_components_stats_check, bkg_t_minus_10 = frames_to_check.get(frame_number)	
				drawBoundingBoxes(connected_components_labels_bg, connected_components_stats_bg, frame_t)
				
				print(frame_number)

				


			bkg_Canny = cv2.cvtColor(bkg_Canny, cv2.COLOR_GRAY2RGB)	
			fgmask_rgb = np.hstack((frame_t, fgmask_rgb, bkg, bkg_Canny, diff))


		cv2.imwrite(os.path.join(sub_dir, frm_file), fgmask_rgb)

		cv2.imshow('frame', fgmask_rgb)

		frame_number += 1
		
		k = cv2.waitKey(1) 
  
		if k  == ord('q'):
			break
     


if __name__ == '__main__':
	main()