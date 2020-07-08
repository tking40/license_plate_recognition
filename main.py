#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:51:52 2020

@author: thomasj.king
"""

#%% Setup
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2 as cv
import sys
from matplotlib import pyplot as plt

import utils
    
#%% Read in file and convert to grayscale
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


args_image_file = './images/license_plate3.png'
args_width = 0.955

image = cv.imread(args_image_file)
gray_temp = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray_temp, (11,11), 0)
# gray = gray_temp

utils.mycvplot(image,gray)

#%% perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged1 = cv.Canny(gray, 50, 100)
# edged1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
# edged1 = utils.auto_canny(gray)
# edged = cv.dilate(255-edged1, None, iterations=3)
# edged = cv.erode(edged2, None, iterations=1)
edged = cv.morphologyEx(edged1, cv.MORPH_CLOSE, None,iterations = 1)

utils.mycvplot(edged1,edged)

#%% find contours in the edge map
(_,cnts,_) = cv.findContours(edged.copy(), cv.RETR_LIST,
	cv.CHAIN_APPROX_SIMPLE)
# ^ Each individual contour is a Numpy array of (x,y) coordinates of 
# boundary points of the object.

#%% Draw Contours
img = cv.drawContours(image.copy(), cnts, -1, (0,255,0), 3)

utils.mycvplot(img)

#%% sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)

#%% loop over the contours individually
# for ix in np.arange(0,len(cnts)):
#     c = cnts[ix]
    
    
# 	# if the contour is not sufficiently large, ignore it
#     if cv.contourArea(c) < 100:
#         print("Ignoring Contour ",ix+1)
#         continue
# 	# compute the rotated bounding box of the contour
#     orig = image.copy()
#     box1 = cv.minAreaRect(c)
#     box2 = cv.boxPoints(box1)
#     box3 = np.array(box2, dtype="int")
    
    
    
    
    
# 	# order the points in the contour such that they appear
# 	# in top-left, top-right, bottom-right, and bottom-left
# 	# order, then draw the outline of the rotated bounding
# 	# box
#     box4 = perspective.order_points(box3)
#     cv.drawContours(orig, [box4.astype("int")], -1, (0, 255, 0), 2)
 
    

 
# 	# draw the object sizes on the image                                    
# # 	cv.putText(orig, "{:.1f}in".format(dimA),
# # 		(int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX,
# # 		0.65, (255, 255, 255), 2)
 
# 	# show the output image
#     utils.mycvplot(orig)
    
#%% Get candidate contours
# plate_aspect_ratio = 4.63
plate_aspect_ratio = 2
plate_aspect_thresh = 5/100 # percent error tolerance on aspect ratio
N_cnts = len(cnts)
candidates = []
aspect_errors = np.array([[]])
for ix in np.arange(0,len(cnts)):
    c = cnts[ix]   
    x,y,w,h = cv.boundingRect(c)
    aspect_ratio = float(w)/h
    # if aspect ratio is off, ignore it
    aspect_error = abs(1 - aspect_ratio/plate_aspect_ratio)
    if  aspect_error > plate_aspect_thresh:
        print("AR: Ignoring Contour ",ix+1)
        continue
    candidates.append(c)
    aspect_errors = np.append(aspect_errors,aspect_error)
    print("Contour ",ix+1,"/",N_cnts," is candidate with aspect ratio: ",aspect_ratio,"--> ",aspect_error*100,"% off")   

utils.plotContours(image, candidates)
#%% Choose best contour and create mask
min_err = 1
for ix in np.arange(0,len(candidates)):
    c = candidates[ix]
    print(min_err)
    if aspect_errors[ix] == 0.0:
        print("contour",ix+1,"has unrealistic aspect ratio")
        continue
    elif aspect_errors[ix] < min_err:
        print(aspect_errors[ix],"<",min_err)
        print("choosing contour",ix+1)
        best_c = c
        min_err = aspect_errors[ix]
    

box = cv.boxPoints(cv.minAreaRect(best_c))

cmin = np.min(box[:,0]).astype('int')
cmax = np.max(box[:,0]).astype('int')
rmin = np.min(box[:,1]).astype('int')
rmax = np.max(box[:,1]).astype('int')

# plate = gray[rmin:rmax,cmin:cmax]
plate_col = image[rmin:rmax,cmin:cmax]
plate = cv.cvtColor(plate_col, cv.COLOR_BGR2GRAY)

utils.mycvplot(plate_col,plate)
#%% Threshold
thresh1 = cv.threshold(plate,127,255,cv.THRESH_BINARY)
# thresh1 = cv.threshold(plate,127,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C)
utils.mycvplot(thresh1[1])

plate_bin = 255 - thresh1[1]

utils.mycvplot(plate_bin)

#%% Segment
# hist = cv.calcHist([image],[0],None,[256],[0,256])
hist = utils.yhist(plate_bin)
plt.figure()
plt.plot(hist)

(_,cnts,_) = cv.findContours(plate_bin.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)

utils.plotContours(plate_col,cnts)

#%% Get inner height data to pick out characters
orig = plate_col.copy()
cnt_height = np.array([[]])
plate_height = plate_col.shape[0]
for ix in np.arange(0,len(cnts)):
    c = cnts[ix]
    x,y,w,h = cv.boundingRect(c)
    if h < plate_height/3: # skip contours too short to be letters
        continue
    cnt_height = np.append(cnt_height,h)
    
med_height = np.median(cnt_height)
#%% Plot characters if close to median height
height_tol = .2
orig = plate_col.copy()
candidates = []
for ix in np.arange(0,len(cnts)):
    c = cnts[ix]
    x,y,w,h = cv.boundingRect(c)
    height_err = abs(1 - h/med_height)
    if height_err > height_tol:
        print("Ignoring Contour ",ix+1)
        continue
    candidates.append(c)
    
    
utils.plotContours(plate_col,candidates)

#%% reshape to mnist images
resized_characters = []
n_pad = 8
for ix in np.arange(0,len(candidates)):
    #x, y is the top left corner, and w, h are the width and height 
    x,y,w,h = cv.boundingRect(candidates[ix])
    
    character = plate_bin[y:y+h,x:x+w]
    character = np.pad(character,((n_pad,n_pad),(n_pad,n_pad)),'constant')
    
    resized_characters.append(cv.resize(character,(28,28),interpolation = cv.INTER_AREA))

utils.mycvplot(*resized_characters)
#%% Classify characters
import pickle
clf = pickle.load(open( "./mnist/myclf.p", "rb" ))

#%%

for char in resized_characters:
    newchar = cv.erode(char,np.ones((2,2)),iterations=1)
    # newchar = char
    utils.mycvplot(newchar)
    print(clf.predict(np.reshape(newchar,[1,784])))

