import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np 
import json 

# # then just type in following

# img = cv2.imread('bowen_2/001_flow.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# '''
# you might have to disable following flags, if you are reading a semantic map/label then because it will convert it into binary map so check both lines and see what you need
# ''' 
# # img = cv2.imread(PATH2EXR) 
 
# print(img.shape)
# for i in range(4): 
# 	print(i,np.min(img[:,:,i]),np.max(img[:,:,i]))

im = cv2.imread('tmp/000.png')
# im_tm1 = cv2.imread('bowen_2/000.png')
# im_tp1 = cv2.imread('bowen_2/0002.png')

with open('tmp/pose_bones.json', 'r') as f:
    # Load the JSON data from the file
    data = json.load(f)
# print(data.keys())

intrinsics = np.array([[560,0,256],[0,560.0,256,],[0,0,1]])
dist_coeffs = np.array([00, 0, 0, 0], dtype=np.float32)
for bone in data['0'].keys():
	p = np.array([data['0'][bone]])

	p = np.dot(p, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32))

	# p[0][-1]*=-1


	pp,_ = cv2.projectPoints(p,(0,0,0),(0,0,0),intrinsics,dist_coeffs)
	print(pp[0][0])
	pp = (int(pp[0][0][0]),int(pp[0][0][1]))
	im = cv2.circle(im, pp, 3, (0,255,0), 2)
cv2.imwrite('tmp.png',im)