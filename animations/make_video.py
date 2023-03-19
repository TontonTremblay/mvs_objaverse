import subprocess
import glob 
import cv2
import numpy as np 

# folders = glob.glob("output/*/")
folders = glob.glob("output_handal_hammers/*/")

for folder in folders:
    # for ob in glob.glob(f"{folder}/*/"):
    print(folder)
    for png in glob.glob(f"{folder}/*.png"):
        # im = cv2.imread(png,cv2.IMREAD_UNCHANGED)
        # f = np.ones([im.shape[0],im.shape[1],3])*255
        # a = (im[:,:,-1]/255.0).reshape([im.shape[0],im.shape[1],1])
        # a = np.concatenate([a,a,a],2)
        # im = im[:,:,:3]
        # # f = f*(a) + (1-a) * im 
        # f[im[:,:,-1]>0] = im[im[:,:,-1]>0]
        im = cv2.imread(png,cv2.IMREAD_UNCHANGED)
        t = np.ones([im.shape[0],im.shape[1],4])*255
        t[im[:,:,-1]>0] = im[im[:,:,-1]>0]
        cv2.imwrite(png, t)
        # raise()
    subprocess.call(['ffmpeg',\
        '-y',\
        '-framerate', "24", \
        '-pattern_type', 'glob', '-i',\
        f"{folder}/*.png", 
        "-c:v", "libx264","-pix_fmt", "yuv420p",\
        f"output_handal_hammers/{folder.split('/')[-2]}.mp4"
        ]) 


mp4s = sorted(glob.glob("output_handal_hammers/*.mp4"))

# # ffmpeg \
# # -i input0.mp4 -i input1.mp4 -i input2.mp4 -i input3.mp4 \
# # -filter_complex \
# # "[0:v][1:v]hstack=inputs=2[top]; \
# # [2:v][3:v]hstack=inputs=2[bottom]; \
# # [top][bottom]vstack=inputs=2[v]" \
# # -map "[v]" \
# # finalOutput.mp4

# tocall = ["ffmpeg"]

# for i, mp4 in enumerate(mp4s):
#   tocall.append('-i')
#   tocall.append(mp4)
#   if i >= 4:
#       break
# tocall.append("-filter_complex")
# tocall.append("[0:v][1:v]hstack=inputs=2[top];")
# tocall.append("[2:v][3:v]hstack=inputs=2[bottom];")
# tocall.append("[top][bottom]vstack=inputs=2[v]")
# tocall.append('-map')
# tocall.append('"[v]"')
# tocall.append("../final.mp4")
# print(tocall)
# subprocess.call(tocall)


from moviepy.editor import *
clips = []
a = []
w = 10
for i, mp4 in enumerate(mp4s):
    if len(a)>=w:
        clips.append(a)
        a = []
    a.append(VideoFileClip(mp4))

while len(a)<w:
    a.append(a[-1])
clips.append(a)
final = clips_array(clips)
final.write_videofile("output.mp4")