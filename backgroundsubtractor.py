import cv2
import numpy as np

def backgroundsubtract(a):
	cap =cv2.VideoCapture(a)
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
	a=input("enter path:")
	if(a.isdigit()==True):
		a=int(a)
	backgroundsubtract(a)
