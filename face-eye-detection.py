import cv2
import numpy as np

def face_eye_detection(path):
	face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
	eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
	cap=cv2.VideoCapture(path)
	while True:
		ret,img = cap.read()
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		faces=face_cascade.detectMultiScale(gray,1.3,5)
		for(x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),3)
			roi_in_gray=gray[y:y+h,x:x+w]
			roi_in_colour=img[y:y+h,x:x+w]
			eyes=eye_cascade.detectMultiScale(roi_in_gray)
			for(eye_x,eye_y,eye_w,eye_h) in eyes:
				cv2.rectangle(roi_in_colour,(eye_x,eye_y),(eye_x+eye_w,eye_y+eye_h),(255,255,255),2)
		cv2.imshow("Detection",img)
		if cv2.waitKey(5) & 0xff==ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__=='__main__':
	path=input("Enter Video path to capture:")
	if(path.isdigit()==True):
		path=int(path)
	face_eye_detection(path)
