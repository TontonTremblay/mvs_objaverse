import os
import argparse

parser = argparse.ArgumentParser(description='Renders glbs')
parser.add_argument(
    '--save_folder', type=str, default='/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/handal_hammer_syn/',
    help='path for saving rendered image')
parser.add_argument(
    '--folder_assets', type=str, default='/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/handal_hammer_assets/',
    help='path to downloaded 3d assets')
parser.add_argument(
    '--blender_root', type=str, default='/home/jtremblay/Desktop/blender-3.2.0-alpha+master.e2e4c1daaa47-linux.x86_64-release/blender',
    help='path to blender executable')
opt = parser.parse_args()



# get all the file
import glob 

# models = sorted(glob.glob(opt.folder_assets +"*.glb"))

# print(model)
for i in range(100):
    os.makedirs(f"{opt.save_folder}/{str(i).zfill(3)}/",exist_ok=True)
    render_cmd = f'{opt.blender_root} -b -P rendering/falling_scene.py -- --folder_assets {opt.folder_assets} --output {opt.save_folder}/{str(i).zfill(3)}/ --views 200 --input_model glb --resolution 400' 
    render_cmd = render_cmd+" > tmp.out"

    print(render_cmd)
    os.system(render_cmd)


