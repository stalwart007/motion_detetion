import cv2
import numpy as np


vid = cv2.VideoCapture('k.mp4')
# fourcc = cv2.VideoWriter_fourcc(*'XVOD')
# out = cv2.VideoWriter('output12.mp4',fourcc,20.0,(640,480))

_,frame1 = vid.read()
_,frame2 = vid.read()

while(vid.isOpened()):
    frame1 = cv2.resize(frame1,(512,512))
    frame2 = cv2.resize(frame2,(512,512))

    diff  = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    guassian_blur = cv2.GaussianBlur(gray,(5,5),0)
    _,thresh = cv2.threshold(guassian_blur,20,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh,None,iterations=5)

    contours,_ = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(frame1,contours,-1,(0,255,0),3)

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if(cv2.contourArea(contour) > 1500):
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow('FEED',frame1)
    frame1 = frame2
    _,frame2 = vid.read()

    k = cv2.waitKey(20)
    if(k == ord('q')):
        break

cv2.destroyAllWindows()
vid.release()
