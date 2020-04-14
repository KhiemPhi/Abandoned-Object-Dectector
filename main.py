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
	diff_dir = '../data/diff/video5/'
	frm_dir = '../data/frames/video5/'
	sub_dir = '../data/subtract/video5/'
	bg_dir = '../data/bg/video5/'    
	detected_dir = '../data/detected/video5'
	

	frm_files = sorted(fnmatch.filter(os.listdir(frm_dir), '*.jpg'))

	frame_0 = cv2.imread(os.path.join(frm_dir, frm_files[0]))
	
	fgbg = cv2.createBackgroundSubtractorMOG2()
	

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	
	total_bg_connected_components_past = 0	
	total_fg_connected_components_past = 0
	frame_number = 2
	check_window = int(0.5 * len(frm_files[1:]))
	frames_to_check = dict()
	diff = np.zeros(frame_0.shape, np.uint8)
	components_area_lower = 10
	components_area_upper = 200
	total_frames = len(frm_files[1:])
	
    

	for frm_file in frm_files[1:]:
		# fgmask = fgbg.apply(frame_0)
		frame_t = cv2.imread(os.path.join(frm_dir, frm_file))
		bkg = fgbg.getBackgroundImage() #---> Get Background Image

		fgmask = fgbg.apply(frame_t) # Find the foreground mask

		#Thresholding : Foreground Mask
		fgmask[fgmask < 128] = 0
		fgmask[fgmask != 0] = 255

		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
		fgmask_rgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB) # Convert to same channel to Stack with frame + bkg image

		# Get Forground Connected Components
		connected_components_labels_fg, connected_components_stats_fg, num_connected_components_fg = findConnectedComponents(fgmask, components_area_lower, components_area_upper)


		# Get Background Connected Components
		if bkg is not None:			
			bkg_Canny = cv2.Canny(bkg, 10, 200)
			connected_components_labels_bg, connected_components_stats_bg, num_connected_components_bg = findConnectedComponents(bkg_Canny, components_area_lower, components_area_upper)
					
			
			# If current bg components is greater than the past bg components then we add the current frame as a suspected frame to our dict
			if ( total_bg_connected_components_past < len(connected_components_labels_bg) ):				
				total_bg_connected_components_past = len(connected_components_labels_bg)
				#print(total_bg_connected_components_past)
				suspected_frame = frame_number + check_window
				frames_to_check.update({suspected_frame: (bkg, frame_number) })
             
			# If current bg components is greater than the past bg components then we add the current frame as a suspected frame to our dict
			'''
			if ( total_fg_connected_components_past <= len(connected_components_labels_fg) ):				
				total_fg_connected_components_past = len(connected_components_labels_fg)
				#print(total_fg_connected_components_past)
				suspected_frame = frame_number + check_window
				frames_to_check.update({suspected_frame: (bkg, frame_number) })
			'''
				


			if (frame_number in frames_to_check): # If the frame we are in is a suspected frame, we check the no. of connected components
				bkg_t_minus_n, frame_past = frames_to_check.get(frame_number)
				diff = cv2.absdiff(bkg_t_minus_n, bkg) 
				diff_canny = cv2.Canny(diff, 10, 100)
				connected_components_labels_diff, connected_components_stats_diff, num_connected_components_diff = findConnectedComponents(diff_canny, components_area_lower, components_area_upper)
				drawBoundingBoxes(connected_components_labels_diff, connected_components_stats_diff, frame_t)
				print(str(frame_past) + " " + str(frame_number))
				cv2.imwrite(os.path.join(diff_dir, frm_file), diff)
				results = np.hstack((frame_t, bkg, bkg_t_minus_n, diff))
				cv2.imwrite(os.path.join(detected_dir, frm_file), results)

			
			fgmask_rgb = np.hstack((frame_t, fgmask_rgb, bkg,  diff))

		if bkg is not None:
			cv2.imwrite(os.path.join(bg_dir, frm_file), bkg)
			

		cv2.imshow('frame', fgmask_rgb)

		frame_number += 1
		
		k = cv2.waitKey(1) 
  
		if k  == ord('q'):
			break
     


if __name__ == '__main__':
	main()