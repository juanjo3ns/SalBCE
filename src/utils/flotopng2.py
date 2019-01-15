################################ Reading flow file ################################
import numpy as np
import cv2
import os

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

flowpath= "/home/dataset/DHF1K/dhf1k_flow"
os.makedirs("/home/dataset/DHF1K/dhf1k_flow")
path = '/home/dataset/DHF1K/dhf1k_flowmotion/'
for folder in range(601,701):
	if not os.path.exists(os.path.join(flowpath, str(folder))):
		os.makedirs(os.path.join(flowpath, str(folder)))
	for img in os.listdir(os.path.join(path,str(folder))):
		if '.flo' in img:
		    flo_file = os.path.join(path,str(folder),img)
		    f = open(flo_file, 'rb')

		    x = np.fromfile(f, np.int32, count=1) # not sure what this gives
		    w = np.fromfile(f, np.int32, count=1) # width
		    h = np.fromfile(f, np.int32, count=1) # height
		    # print 'x %d, w %d, h %d flo file' % (x, w, h)

		    data = np.fromfile(f, np.float32) # vector

		    # data_2D = np.reshape(data, newshape=(384,512,2)); # convert to x,y - flow
		    data_2D = np.reshape(data, newshape=(192,256,2)); # convert to x,y - flow
		    x = data_2D[...,0]; y = data_2D[...,1];


		    ################################ visualising flow file ################################
		    mag, ang = cv2.cartToPolar(x,y)
		    hsv = np.zeros_like(x)
		    hsv = np.array([ hsv,hsv,hsv ])
		    # hsv = np.reshape(hsv, (384,512,3)); # having rgb channel
		    hsv = np.reshape(hsv, (192,256,3)); # having rgb channel
		    hsv[...,1] = 255; # full green channel
		    hsv[...,0] = ang*180/np.pi/2 # angle in pi
		    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) # magnitude [0,255]
		    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
		    bgr = draw_hsv(data_2D)
		    cv2.imwrite(os.path.join(flowpath, str(folder), img.split('.')[0]+'.png'),bgr)
