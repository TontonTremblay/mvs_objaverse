import os
import argparse

parser = argparse.ArgumentParser(description='Renders glbs')
parser.add_argument(
    '--save_folder', type=str, default='/Users/jtremblay/code/mvs_objaverse/output/falling_mugs/',
    help='path for saving rendered image')
parser.add_argument(
    '--folder_assets', type=str, default='/Users/jtremblay/code/mvs_objaverse/assets/objaverse/mug/',
    help='path to downloaded 3d assets')
parser.add_argument(
    # '--blender_root', type=str, default='/Applications/Blender.app/Contents/MacOS/Blender',
    '--blender_root', type=str, default='/Applications/Blender_34.app/Contents/MacOS/Blender',
    help='path to blender executable')
opt = parser.parse_args()



# get all the file
import glob 

# models = sorted(glob.glob(opt.folder_assets +"*.glb"))

# print(model)


for i in range(10): 
    output_folder = f"{opt.save_folder}/{str(i).zfill(3)}"
    render_cmd = f'{opt.blender_root} -b -P rendering/falling_scene.py -- --folder_assets {opt.folder_assets} --output {output_folder} --assets_hdri /Users/jtremblay/code/mvs_objaverse/assets/dome_hdri_haven/ --asset_textures /Users/jtremblay/code/mvs_objaverse/assets/cco_textures/ --save_tmp_blend /Users/jtremblay/code/mvs_objaverse/tmp.blend --distractors 0 --views 4 --input_model glb --resolution 512' 
    # render_cmd = render_cmd + ' > tmp.out'

    print(render_cmd)
    os.system(render_cmd)


