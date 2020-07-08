import os
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


## Filesystem functions
def ls():
    curdir = os.getcwd()
    print('Current Directory: ' + curdir)
    print('\\\\')
    f = os.listdir(curdir)
    for filename in f:
        print(filename)
    print('\\\\')

def isimported(file):
    return file in sys.modules


## Syntax functions
def step(start,step,stop):
    return np.arange(start,stop+step,step)


def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv :", arg)

## Plotting Utils
def mycvplot(*argv):
    # Note: assumes that a 2D input is a grayscale image
    # Also, the imutils package has a function that can handle conversion:
    # plt.imshow(imutils.opencv2matplotlib(cactus))
    # But I can do this with the cv2.cvtColor, so I'll avoid using imutils
    # unless I find I need to
    if any('SPYDER' in name for name in os.environ):
        for arg in argv:
            # if isinstance(arg,list):
            #     mycvplot(arg)
            plt.figure()
            if len(arg.shape) > 2:
                plt.imshow(cv.cvtColor(arg, cv.COLOR_BGR2RGB))
            else:
                plt.imshow(arg,cmap='gray')
            plt.show()
    else:
        for arg in argv:
            cv.imshow("img", arg)
            cv.waitKey(0)
            # cv2.destroyAllWindows()

## Image Utils         
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)
	# return the edged image
	return edged

def plotContours(image,cnts):
    orig = image.copy()
    for ix in np.arange(0,len(cnts)):
        c = cnts[ix]
        x,y,w,h = cv.boundingRect(c)    
    
        cv.rectangle(orig, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    mycvplot(orig)
    
def imresize(img,sf):
    dim = tuple(np.multiply(sf,img.shape[:2]).astype("int"))
    # resize image here
    return cv.resize(img,(dim[1],dim[0]),interpolation = cv.INTER_AREA)

def yhist(image,ax=0):
    # Compute 1-dimensional histogram of intensity values for image
    # default is columnwise
    N = len(image.shape)
    if N < 3:
        return np.sum(image,axis=ax)
    else:
        out = []
        for i in range(3):
            print(i)
            out.append(np.sum(image[:,:,i],axis=ax))
        return out