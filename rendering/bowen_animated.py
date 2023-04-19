import sys
with open("path.txt") as file:
    paths = eval(file.read())
for p in paths:
    sys.path.insert(0,p)
import argparse, sys, os, math, re
import bpy
from mathutils import Vector, Matrix
import mathutils
import numpy as np
import json 
import random 
import glob 
import threading


from yourdfpy import URDF
import yourdfpy
import random 
import pyrr 
import scipy 
from utils import * 


##### CLEAN BLENDER SCENES ##### 
bpy.ops.object.delete()
# bpy.ops.objects['Light'].delete()
bpy.data.objects['Light'].select_set(True)
bpy.ops.object.delete()

RESOLUTION = 512
##### SET THE RENDERER ######
render = bpy.context.scene.render
render.engine = "CYCLES"
render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
render.image_settings.file_format = 'PNG'  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = RESOLUTION
render.resolution_y = RESOLUTION
render.resolution_percentage = 100
bpy.context.scene.cycles.filter_width = 0.01
# bpy.context.scene.render.film_transparent = True

bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.diffuse_bounces = 1
bpy.context.scene.cycles.glossy_bounces = 1
bpy.context.scene.cycles.transparent_max_bounces = 3
bpy.context.scene.cycles.transmission_bounces = 3
bpy.context.scene.cycles.samples = 32
bpy.context.scene.cycles.use_denoising = True


##### LOAD THE ANIMATED SCENE #####
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0
bpy.context.scene.render.film_transparent = True

bpy.ops.import_scene.fbx( filepath = "/Users/jtremblay/Downloads/abe_CoverToStand/abe_CoverToStand.fbx" )

random.seed(100101)
np.random.seed(100101)


# get keyframes of object list
def get_keyframes(obj_list):
    keyframes = []
    for obj in obj_list:
        anim = obj.animation_data
        if anim is not None and anim.action is not None:
            for fcu in anim.action.fcurves:
                for keyframe in fcu.keyframe_points:
                    x, y = keyframe.co
                    if x not in keyframes:
                        keyframes.append((math.ceil(x)))
    return keyframes

# get all selected objects
selection = bpy.context.selected_objects


# get all frames with assigned keyframes
keys = get_keyframes(selection)
bpy.context.scene.frame_end = keys[-1]

bpy.ops.object.empty_add(radius=0.05,location=(0,0,0))
add_light_under(bpy.context.selected_objects[0],-3,power=500)

bpy.ops.object.empty_add(radius=0.05,location=(2,2,0))
add_light_under(bpy.context.selected_objects[0],-1,power=500)


####### make camera poses #########

look_at_trans = []
global DATA_EXPORT


#### set up the camera 
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

cam = scene.objects['Camera']
cam.location = (1.5477,1.10554, 2.20848)  # radius equals to 1
cam.rotation_euler.x = 59.5244 *np.pi/180
cam.rotation_euler.y = 0.716649 *np.pi/180
cam.rotation_euler.z = 136.523 *np.pi/180 

cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

bpy.context.scene.render.resolution_x = RESOLUTION
bpy.context.scene.render.resolution_y = RESOLUTION

make_segmentation_scene()

bpy.ops.wm.save_as_mainfile(filepath=f"/Users/jtremblay/code/mvs_objaverse/bowen.blend")

for i in range(keys[-1]+1):
    render_single_image(
        frame_set = i,
        look_at_data=None,
        path = "/Users/jtremblay/code/mvs_objaverse/tmp/",
        resolution = RESOLUTION,
        )

# print(look_at_trans[0])




