import os
import cv2
import fnmatch
import numpy as np
import imutils
import glob
import argparse

def compareBoundingBoxes(labels, stats):
	return labels


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
	#connected_components_labels = list(filter(lambda x: connected_components_stats[x, cv2.CC_STAT_AREA] > components_area_lower and connected_components_stats[x, cv2.CC_STAT_AREA] < components_area_upper , connected_components_labels))
	return connected_components_labels, connected_components_stats, num_connected_components

def clearDirectories (path):
	files = glob.glob(path + '*')
	for f in files:
		os.remove(f)

def getBrighterFrame(frame1, frame2):
	frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
	frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
	print(frame1[...,2])
	print(frame2[...,2])

def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-w", "--check_window", required=False,
	help="check window")
	parser.add_argument("-al", "--area_lower", required=False,
	help="component area lower limit")
	parser.add_argument("-au", "--area_upper", required=False,
	help="component area upper limit")
	parser.add_argument("-cl", "--canny_lower", required=False,
	help="canny lower threshold")
	parser.add_argument("-cu", "--canny_upper", required=False,
	help="canny upper threshold")
	parser.add_argument("-d", "--dataset", required=False, help = "dataset to run videos, default ABODA")
	parser.add_argument("-v", "--video", required=False, help = "default video to test, default ABODA 1")



	args = vars(parser.parse_args())


	#video_lower_index = 1
	#video_upper_index = len([name for name in os.listdir() if os.path.isfile(os.path.join(DIR, name))])

	
	video_number_for_path = str(args["video"]) if args["video"] else "1"	


	dataset = str(args["dataset"]) if args["dataset"] else "ABODA"


	diff_dir = '../data/diff/' +  dataset +  '/video' + video_number_for_path + '/'
	frm_dir = '../data/frames/' +  dataset +  '/video' + video_number_for_path + '/'
	sub_dir = '../data/subtract/' +  dataset +  '/video' + video_number_for_path + '/'
	bg_dir = '../data/bg/' +  dataset +  '/video' + video_number_for_path + '/'    
	detected_dir = '../data/detected/' +  dataset +  '/video' + video_number_for_path + '/'
	stack_dir = '../data/stack/' +  dataset +  '/video' + video_number_for_path + '/'

	frm_files = sorted(fnmatch.filter(os.listdir(frm_dir), '*.jpg'))

	frame_0 = cv2.imread(os.path.join(frm_dir, frm_files[0]))
	
	fgbg = cv2.createBackgroundSubtractorMOG2()	

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	
	total_bg_connected_components_past = 0	
	total_fg_connected_components_past = 0
	frame_number = 2

	total_frames = len(frm_files[1:])
	
	check_window = int(args["check_window"]) if args["check_window"] else 500  #int(0.9 * len(frm_files[1:])) 
	frames_to_check = dict()
	diff = np.zeros(frame_0.shape, np.uint8)
	diff_canny = np.zeros(frame_0.shape, np.uint8)

	components_area_lower = int(args["area_lower"]) if args["area_lower"] else 0
	components_area_upper = int(args["area_upper"]) if args["area_upper"] else 1000
	
	canny_threshold_lower = int(args["canny_lower"]) if args["canny_lower"] else 10 
	canny_threshold_upper = int(args["canny_upper"]) if args["canny_upper"] else 200	
	
	#Clearing the Directories With Detected Frames
	clearDirectories(detected_dir)
    

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


			
			bkg_Canny = cv2.Canny(bkg, canny_threshold_lower, canny_threshold_upper)
			
			connected_components_labels_bg, connected_components_stats_bg, num_connected_components_bg = findConnectedComponents(bkg_Canny, components_area_lower, components_area_upper)
						
			
             
			# If current bg components is greater than the past bg components then we add the current frame as a suspected frame to our dict
			
			

			if ( total_bg_connected_components_past < len(connected_components_labels_bg) ):	
				
				
				#print(total_fg_connected_components_past)
				suspected_frame = frame_number + check_window

				if (suspected_frame > total_frames):
					suspected_frame = total_frames

				range_to_check = range(suspected_frame-50, suspected_frame+50)
				add_to_dict = True
				frame_to_update = 0
				for i in range_to_check:
					if i in frames_to_check:
						add_to_dict = False
						frame_to_update = i
				
				if (add_to_dict == True):
				    frames_to_check.update({suspected_frame: [bkg, frame_t] })				    
				elif (frame_to_update != 0 ):
				 	frames_to_check.update({frame_to_update: [bkg, frame_t]})
				 	
				
			
				


			if (frame_number in frames_to_check  ): # If the frame we are in is a suspected frame, we check the no. of connected components
				
				bkg_t_minus_n,	frame_past = frames_to_check.get(frame_number)
				bkg_gray = cv2.cvtColor(bkg, cv2.COLOR_BGR2GRAY)
				bkg_past_gray = cv2.cvtColor(bkg_t_minus_n, cv2.COLOR_BGR2GRAY)
				
				# Histogram Equalization to deal with Lighting Changes
				bkg_equal = cv2.equalizeHist(bkg_gray)
				bkg_past_equal = cv2.equalizeHist(bkg_past_gray)


				diff = cv2.absdiff(bkg_equal, bkg_past_equal) 
				

				diff_canny = cv2.Canny(diff, canny_threshold_lower, canny_threshold_upper)	
					

				
				connected_components_labels_diff, connected_components_stats_diff, num_connected_components_diff = findConnectedComponents(diff_canny, components_area_lower, components_area_upper)
				drawBoundingBoxes(connected_components_labels_diff, connected_components_stats_diff, frame_t)
								


				cv2.imwrite(os.path.join(diff_dir, frm_file), diff)
				diff_canny = cv2.cvtColor(diff_canny, cv2.COLOR_GRAY2BGR)
				diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
				bkg_equal = cv2.cvtColor(bkg_equal, cv2.COLOR_GRAY2BGR)
				bkg_past_equal = cv2.cvtColor(bkg_past_equal, cv2.COLOR_GRAY2BGR)
				

				results = np.hstack((frame_t, frame_past, bkg_t_minus_n, bkg, diff, diff_canny))
				cv2.imwrite(os.path.join(detected_dir, frm_file), results)



			
			fgmask_rgb = np.hstack((frame_t, fgmask_rgb, bkg,  diff, diff_canny))

			
			total_bg_connected_components_past = len(connected_components_labels_bg) + 0

			    
			


		if bkg is not None:
			cv2.imwrite(os.path.join(bg_dir, frm_file), bkg)
			cv2.imwrite(os.path.join(stack_dir, frm_file), fgmask_rgb)
			

		cv2.imshow('frame', fgmask_rgb)

		frame_number += 1
		
		k = cv2.waitKey(1) 
  
		if k  == ord('q'):
			break
     


if __name__ == '__main__':
	main()