import os
import argparse

parser = argparse.ArgumentParser(description='Renders glbs')
parser.add_argument(
    '--save_folder', type=str, default='/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/objaverse_renders/',
    help='path for saving rendered image')
parser.add_argument(
    '--folder_assets', type=str, default='/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/objaverse',
    help='path to downloaded 3d assets')
parser.add_argument(
    '--blender_root', type=str, default='/home/jtremblay/Desktop/blender-3.2.0-alpha+master.e2e4c1daaa47-linux.x86_64-release/blender',
    help='path to blender executable')
opt = parser.parse_args()



# get all the file
import glob 
data = sorted(glob.glob(f"{opt.folder_assets}/*/"))


for folder in data:
    paths = glob.glob(folder+'/*/')
    for path in paths:
        # path = data[-5]
        # raise()
        path = sorted(glob.glob(path + "/*.glb"))
        if len(path) == 0: 
            continue
        path = path[0]
        name = path.split('/')[-3] + "_"+path.split('/')[-1].replace(".glb","")
        # print(name)
        # raise()

        if os.path.exists(os.path.join(os.path.abspath(opt.save_folder), name)):
            print("rendered")
            print(os.path.join(os.path.abspath(opt.save_folder), name))

            continue
        # raise()
        render_cmd = '%s -b -P rendering/render_blender.py -- --obj %s --output %s --outf_name %s --views 100 --resolution 256' % (
            opt.blender_root, path, opt.save_folder,name
        )
        print(render_cmd)
        os.system(render_cmd)
    #     break
    # break