import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import glob
import json 
import numpy as np 
import colorsys
import random 

random.seed(100)
path = 'tmp/000.json'
im = cv2.imread(path.replace('json','png'))
seg = cv2.imread(path.replace('.json','_seg.exr'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
seg = (seg*255).astype(int)

with open(path, 'r') as f:
    # Load the JSON data
    data = json.load(f)

# generate the handle 
colors_2_keep = []

for obj in data['objects']:
    if 'handle' in obj['name'] :
        print(obj['name'],obj['parent'])
        c = obj['color_seg'].copy()
        for i in range(3): 
            c[i] = int(c[i]*255) 
        colors_2_keep.append(c)

im_black = np.zeros([im.shape[0],im.shape[1],3])

seg = np.asarray(seg)
# seg = seg.reshape((width,height,4))[:,:,:3]
unique_pixels = np.vstack({tuple(r) for r in seg.reshape(-1,3)})
# unique_pixels = (unique_pixels*255).astype(int)

# print(unique_pixels)
# print('-----------')
# print(colors_2_keep)

for c in colors_2_keep:
    im_black[seg == c] = 255

cv2.imwrite('handle.png', im_black)

def get_color():
    c = colorsys.hsv_to_rgb(
        random.uniform(0,255)/255, 
        random.uniform(200,255)/255, 
        random.uniform(200,255)/255
        )
    cc = []
    for i in range(3): 
        cc.append(int(c[i]*255))    
    # return cc
    return np.array(cc)

# Generate the parts segmentation
mapping_c_2_p = []
parent_color = {}
for obj in data['objects']:
    if not 'color_seg' in obj or obj['visibility']==0:
        continue
    c = obj['color_seg']
    for i in range(3): 
        c[i] = int(c[i]*255) 

    if not obj['parent'] in parent_color:
        parent_color[obj['parent']] = get_color()
    c_parent = parent_color[obj['parent']]

    # if 'heater' in obj['name'] :
    # print(obj['name'],obj['parent'],parent_color[obj['parent']],obj['visibility'])

    mapping_c_2_p.append([c,c_parent,obj['name']])

im_black = np.zeros([im.shape[0],im.shape[1],3])

for ic, c in enumerate(mapping_c_2_p):
    print(c)
    # print(seg == c[1])
    im_ones = np.ones([im.shape[0],im.shape[1],3])
    for i in range(3):
        im_ones[:,:,i]*=c[1][i]
        # im_ones[:,:,i]*=0

    # mask = seg == c[0]
    mask = cv2.inRange(seg, np.array(c[0]), np.array(c[0]))
    print(np.max(mask))
    im_black[mask>0] = c[1]

    # if ic == 11:
    #     im_black = np.zeros([im.shape[0],im.shape[1],3]) 
    #     im_ones = np.ones([im.shape[0],im.shape[1],3])
    #     for i in range(3):
    #         im_ones[:,:,i]*=c[0][i]
    #     im_black[mask] = im_ones[mask]

    #     cv2.imwrite('tmp_.png',im_black)

    #     break
# im_black = cv2.cvtColor(im_black, cv2.COLOR_BGR2RGB)
print(parent_color)
cv2.imwrite('parent.png', im_black)
cv2.imwrite('seg_.png',seg)







