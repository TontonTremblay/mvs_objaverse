import os
import argparse
import sys


with open("path.txt", "w") as file:
    file.write(str(sys.path))

parser = argparse.ArgumentParser(description='Renders glbs')
parser.add_argument(
    '--blender_root', type=str, default='/Applications/Blender_34.app/Contents/MacOS/Blender',
    help='path to blender executable')

opt = parser.parse_args()




# render_cmd = f'PYTHONPATH=/Users/jtremblay/miniconda3/bin/python {opt.blender_root} -b --python-use-system-env -P rendering/urdf_scene.py -- ' 
render_cmd = f'PYTHONPATH=/Users/jtremblay/miniconda3/bin/python {opt.blender_root} -b --python-use-system-env -P rendering/bowen_animated.py -- ' 

# render_cmd = render_cmd + ' > tmp.out'

print(render_cmd)
os.system(render_cmd)


