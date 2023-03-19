import cv2 
import numpy as np 
import glob 
import subprocess 
import math 
imgs = []

for folder in glob.glob("output/*/"):
    for obj in glob.glob(folder+"/*/"):
        objs = sorted(glob.glob(obj+"/*.png"))
        imgs.append(objs[-30])

imgs_loaded = []

for img in imgs:
    im = cv2.imread(img,cv2.IMREAD_UNCHANGED)
    t = np.ones([im.shape[0],im.shape[1],4])*255
    t[im[:,:,-1]>0] = im[im[:,:,-1]>0]
    imgs_loaded.append(t)
imgs = imgs_loaded
w = 7
# print(len(imgs)/w)
# print(len(imgs)//w)
# raise()
im = np.ones([
        int(imgs[0].shape[0]*math.ceil(len(imgs)/w)),
        imgs[0].shape[0]*w,
        4
    ])*255
row = -1 
for i_img, img in enumerate(imgs):
    h =  i_img % w
    if h ==0 : 
        row += 1
    print(h,row)
    print('h',imgs[0].shape[0]*row,imgs[0].shape[0]*(row+1))
    print('w',imgs[0].shape[1]*h,imgs[0].shape[1]*(h+1))
    print(im.shape)
    im[
        imgs[0].shape[0]*row:imgs[0].shape[0]*(row+1),
        imgs[0].shape[1]*h:imgs[0].shape[1]*(h+1),
        :
        ] = img

# remove the white 

w = 3
for i in range(im.shape[0]-1-w,-1+w,-1):
    if len(np.unique(im[i:i+w,:,:]))==1:
        # wi = im.shape[0]-i-w
        # im[i:wi,:,:] = im[i-w:wi+w,:,:]
        im = np.delete(im, i,0)
for i in range(im.shape[1]-1-w,-1+w,-1):
    if len(np.unique(im[:,i:i+w,:]))==1:
        # wi = im.shape[0]-i-w
        # im[i:wi,:,:] = im[i-w:wi+w,:,:]
        im = np.delete(im, i,1)
cv2.imwrite('tmp.png',im)