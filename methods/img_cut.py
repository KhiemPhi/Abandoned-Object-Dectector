import cv2
import numpy as np

def main():
	file_path = '../img1.jpg'
	img = cv2.imread(file_path)
	h, w, c = img.shape
	w = w // 3
	

	while True:
		imgCenter = img[0:h, 0:w]
		cv2.imshow('frame', imgCenter)
		k = cv2.waitKey(1)
		print(img.shape)
		print(imgCenter.shape)
		cv2.imwrite('../right.jpg', imgCenter)
		if k == ord('q'):
			break

if __name__ == '__main__':
	main()