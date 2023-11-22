import os
import argparse

parser = argparse.ArgumentParser(description='Renders glbs')
parser.add_argument(
    '--save_folder', type=str, default='/Users/jtremblay/code/mvs_objaverse/output/try2/',
    help='path for saving rendered image')
parser.add_argument(
    '--folder_assets', type=str, default='/Users/jtremblay/code/mvs_objaverse/assets/fix_hammers/00608.glb',
    help='path to downloaded 3d assets')
parser.add_argument(
    '--blender_root', type=str, default='/Applications/Blender.app/Contents/MacOS/Blender',
    help='path to blender executable')
opt = parser.parse_args()



# get all the file
import glob 

def find_glb_files(root_dir):
    glb_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".glb"):
                glb_file_path = os.path.join(dirpath, filename)
                glb_files.append(glb_file_path)
    return glb_files

root_dir = os.getcwd()
glb_files = find_glb_files('assets/objaverse/mug/')

# models = sorted(glob.glob(opt.folder_assets +"*.glb"))

# print(model)
for glb_file in glb_files:
    name = glb_file.split('/')[-1].replace('.glb','')
    outf = f"{opt.save_folder}/{name}/"
    os.makedirs(outf,exist_ok=True)

    render_cmd = f'{opt.blender_root} -b -P rendering/render_blender.py -- --obj {glb_file} --output {outf} --assets_hdri /Users/jtremblay/code/mvs_objaverse/assets/dome_hdri_haven/ --input_model glb --resolution 800 --views 1' 
    # render_cmd = render_cmd + ' > tmp.out'

    print(render_cmd)
    os.system(render_cmd)
    # raise()

