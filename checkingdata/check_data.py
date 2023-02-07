import cv2
import glob 
import subprocess 
import os
import argparse
import numpy as np 

parser = argparse.ArgumentParser(description='Check data rendered, create a txt with which one to keep and loose')
parser.add_argument(
    '--path', 
    type=str, 
    default='/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/objaverse_renders/',
    help='path to images')

opt = parser.parse_args()


folders = sorted(glob.glob(opt.path + "*/"))

keep = []
not_keep = [] 

i_folder = 0 
i_img = 0

i_t = 0
while True: 
    i_t += 1
    data_imgs = sorted(glob.glob(folders[i_folder] + "/*.png"))

    img_to_show = cv2.imread(data_imgs[i_img])
    img_to_show = cv2.resize(img_to_show, 
        (img_to_show.shape[0]*2,img_to_show.shape[1]*2))
    cv2.imshow('Original Image', img_to_show)

    # timer    
    if i_t % 100 == 0 :
        i_img += 1
        if i_img >= len(data_imgs)-1:
            i_img = 0 

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('k'): #keep 
        keep.append(folders[i_folder].split("/")[-2])
        i_folder += 1
    if key == ord('d'): #DONT
        not_keep.append(folders[i_folder].split("/")[-2])
        i_folder += 1
        
    if key == ord('q') or i_folder >= len(folders)-1:
        cv2.destroyAllWindows()
        break

with open("keep.txt", "w") as text_file:
    text_file.write(str(keep))
with open("not_keep.txt", "w") as text_file:
    text_file.write(str(not_keep))
