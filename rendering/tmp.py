import cv2 
import numpy as np 
import sys
print(sys.path)

im = cv2.imread('output/tmp/000_seg.exr',cv2.IMREAD_UNCHANGED)

print(np.unique(im,axis=2))