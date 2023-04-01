import os
import argparse

parser = argparse.ArgumentParser(description='Renders glbs')
parser.add_argument(
    '--save_folder', type=str, default='/data/handal_dataset_syn/handal_dataset_hammers/',
    help='path for saving rendered image')
parser.add_argument(
    '--folder_assets', type=str, default='/data/handal_dataset_syn/hammer_syn_models/aligned_obj/',
    help='path to downloaded 3d assets')
parser.add_argument(
    '--blender_root', type=str, default='/home/andrewg/Downloads/blender-3.4.0-linux-x64/blender',
    help='path to blender executable')
opt = parser.parse_args()



# get all the file
import glob 

# models = sorted(glob.glob(opt.folder_assets +"*.glb"))

# print(model)

render_cmd = f'{opt.blender_root} -b -P rendering/falling_scene.py -- --folder_assets {opt.folder_assets} --output {opt.save_folder}  --save_tmp_blend /home/andrewg/mvs_objaverse/tmp.blend --distractors 0 --views 10 --input_model obj --resolution 512' 
# render_cmd = render_cmd + ' > tmp.out'

print(render_cmd)
os.system(render_cmd)


