import cv2
import numpy as np

def backgroundsubtract(path):
	cap =cv2.VideoCapture(path)
	bgsubs = cv2.createBackgroundSubtractorMOG2()
	while True:
		ret,frame =cap.read()
		mask = bgsubs.apply(frame)
		frame=cv2.bitwise_and(frame,frame,mask=mask)
		cv2.imshow('Background Removed' , frame)
		if cv2.waitKey(5) & 0xFF==ord('q'):
	 		break;

	cap.release()
	cv2.destroyAllWindows()

if __name__=="__main__":
	path=input("enter path:")
	if(path.isdigit()==True):
		path=int(path)
	backgroundsubtract(path)
