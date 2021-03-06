import cv2
import numpy as np
import matplotlib.pyplot as plt
def featurematch(imgpath,tofindpath,n):
	img = cv2.imread(imgpath,1)
	tofind =cv2.imread(tofindpath,1)
	orb = cv2.ORB_create()
	kp1, desc1 = orb.detectAndCompute(img,None)
	kp2, desc2 = orb.detectAndCompute(tofind,None)
	bm=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)
	matches=bm.match(desc1,desc2)	
	matches =  sorted(matches,key =lambda x:x.distance)
	opimg = cv2.drawMatches(img,kp1,tofind,kp2,matches[:n],None,flags=2)
	plt.imshow(opimg)
	plt.show()

if __name__=='__main___':	
	a=input("Enter image path:")	
	b=input("Enter image to match with:")
	n=int(input("Enter number of Key points to be matched:"))
	featurematch(a,b,n)
