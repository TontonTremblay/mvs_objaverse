import os
import argparse

parser = argparse.ArgumentParser(description='Renders glbs')
parser.add_argument(
    '--save_folder', type=str, default='/home/jtremblay/code/mvs_objaverse/output/',
    help='path for saving rendered image')
parser.add_argument(
    '--folder_assets', type=str, default='/home/jtremblay/Downloads/meshes_for_john/release/',
    help='path to downloaded 3d assets')
parser.add_argument(
    '--blender_root', type=str, default='/home/jtremblay/Desktop/blender-3.2.0-alpha+master.e2e4c1daaa47-linux.x86_64-release/blender',
    help='path to blender executable')
opt = parser.parse_args()



# get all the file
import glob 
data = sorted(glob.glob(f"{opt.folder_assets}/*/"))

for path in data:
    # path = data[-5]
    if not 'drill' in path:
        continue
    print('hey')
    type_ = path.split("/")[-2]
    models = sorted(glob.glob(path + "/models/*.ply"))

    for model in models:
        print(model)
        if "handle.ply" in model or "not" in model:
            continue
        model_name = model.split("/")[-1].replace('.ply', "")
        render_cmd = '%s -b -P rendering/render_blender.py -- --obj %s --output %s --views 100 --input_model ply --resolution 300 > tmp.out' % (
            opt.blender_root, model, opt.save_folder + "/"+type_+"/"+model_name
        )
        print(render_cmd)
        os.system(render_cmd)
        break
    break