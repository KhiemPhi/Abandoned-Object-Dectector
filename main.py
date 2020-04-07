import os
import cv2
import fnmatch
import numpy as np


def main():
	frm_dir = './data/frames/video1/'
	sub_dir = './data/subtract/video1/'

	frm_files = sorted(fnmatch.filter(os.listdir(frm_dir), '*.jpg'))

	frame_0 = cv2.imread(os.path.join(frm_dir, frm_files[0]))
	fgbg = cv2.createBackgroundSubtractorMOG2()
	# fgbg2 = cv2.createBackgroundSubtractorMOG2()

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	# fgmask = fgbg.apply(frame_0)

	for frm_file in frm_files[1:]:
		# fgmask = fgbg.apply(frame_0)
		frame_t = cv2.imread(os.path.join(frm_dir, frm_file))
		bkg = fgbg.getBackgroundImage()
		# if bkg is not None:
		# 	print(bkg.shape)
		fgmask = fgbg.apply(frame_t)
		fgmask[fgmask < 128] = 0
		fgmask[fgmask != 0] = 255

		output = cv2.connectedComponentsWithStats(fgmask, 4, cv2.CV_8S)

		num_labels = output[0]
		# The second cell is the label matrix
		for l in range(1, num_labels):
			if output[2][l, -1] < 400:
				fgmask[output[1] == l] = 0

		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
		fgmask_rgb = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2RGB)
		# bkg = cv2.cvtColor(bkg,cv2.COLOR_GRAY2RGB)
		if bkg is not None:
			fgmask_rgb = np.hstack((frame_t, fgmask_rgb, bkg))

		cv2.imwrite(os.path.join(sub_dir, frm_file), fgmask_rgb)

		cv2.imshow('frame', fgmask_rgb)
		k = cv2.waitKey(1) 
  
		if k  == ord('q'):
			break
     


if __name__ == '__main__':
	main()