import cv2
import numpy as np 
import time
import glob 

imgs = sorted(glob.glob('output/*.exr'))

imgs_depth = []
depth_max = 500
for im in imgs:
    im = cv2.imread(im, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    # im = im[:,:,0]

    im[im>depth_max] = depth_max
    # for i in range(3):
    #     print(np.min(im[:,:,i]),np.max(im[:,:,i]))
    im /= np.max(im)

    im *= 255
    imgs_depth.append(im)
i = -1 
while True: 
    i += 1
    if i>=len(imgs_depth):
        i = 0 
    cv2.imshow('frame', imgs_depth[i])  # display the frame
    key = cv2.waitKey(1)  
    
    if key == ord('q'):  
        break
    time.sleep(0.1)
