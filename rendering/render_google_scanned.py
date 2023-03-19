import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Renders glbs')
parser.add_argument(
    '--save_folder', type=str, default='/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/google_scanned_blender_mvs/',
    help='path for saving rendered image')
parser.add_argument(
    '--folder_assets', type=str, default='/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/google_scanned/',
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
    name = path.split("/")[-2]
    print(name)
    path = path + "/meshes/model.obj"
    subprocess.call(["cp",path.replace("/meshes/model.obj","materials/textures/texture.png"),path.replace("model.obj",'')])

    save_folder = opt.save_folder + "/"+ name + "/"
    render_cmd = '%s -b -P rendering/render_blender.py -- --obj %s --input_model obj --output %s --views 100 --resolution 256 > tmp.out' % (
        opt.blender_root, path, save_folder
    )
    print(render_cmd)
    os.system(render_cmd)
    # break