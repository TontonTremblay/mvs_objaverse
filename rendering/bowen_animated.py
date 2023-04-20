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



# get the active object
obj = bpy.context.active_object
# get the animation data and fcurves from the object
# the fcurves are a good way to get direct access to the keyframes
curves = obj.animation_data.action.fcurves

#loop over every fcurve
for c in curves:
    # get the keyframes for every fcurve
    keyframes = c.keyframe_points
    #loop over every keyframe and divide their x coordinate by a scale factor
    for kf in keyframes:
        #this is the number to scale by, 
        #if  you want to scale them all down by half then you'd divide by two
        scale_factor = 0.25
        # co is the keyframe's coordinate attribute
        # we only want to scale them on the x-axis (time) 
        # not on the y (amplitude)
        # /= is a shorthand for saying kf.co.x = kf.co.x / 2
        kf.co.x /= scale_factor




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

bpy.context.view_layer.update()


##### get bone poses #####
for obj in bpy.data.objects:
    obj.select_set(False)

bpy.data.objects["Armature"].select_set(True)
armature = bpy.context.selected_objects[0]

frames = {}
cam = bpy.context.scene.objects['Camera']

for f in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end+1):
    bpy.context.scene.frame_set(f)
    # print("Frame " + str(f) + ": ")
    data = {}
    for boneN in armature.pose.bones.keys():
        # This can be head, tail, center...
        # More info: https://docs.blender.org/api/current/bpy.types.PoseBone.html?highlight=bpy%20bone#bpy.types.PoseBone.bone
        bone = armature.pose.bones[boneN]
        # print(bone.matrix)
        # print(bone.tail)
        # print(bone.head)

        # Since Blender 2.8 you multiply matrices with @ not with *
        # bonePos = armature.matrix_world @ bone.matrix
        bonePos = armature.matrix_world @ bone.matrix

        # add empty 
        # bpy.ops.object.empty_add(location=[bonePos[0][-1],bonePos[1][-1],bonePos[2][-1]])

        # rt = cam.convert_space(matrix=bonePos, to_space='LOCAL')
        rt = cam.matrix_world.inverted() @ bonePos
        # bpy.ops.object.empty_add(location=[rt[0][-1],rt[1][-1],rt[2][-1]])

        # print(boneN,bonePos)
        # print(rt)
        # continue
        # raise()
        data[boneN] = (rt[0][-1],rt[1][-1],rt[2][-1])

        #This may be useful to visualize the location that you have got
        #bpy.ops.mesh.primitive_cube_add(size = 0.1, location=bonePos)
        #armature.pose.bones[boneN]

    # break
    frames[f-1] = data
    # break

with open("/Users/jtremblay/code/mvs_objaverse/tmp/pose_bones.json", 'w+') as fp:
    json.dump(frames, fp, indent=4, sort_keys=True)





##### rendering set up #####

make_segmentation_scene()

bpy.ops.wm.save_as_mainfile(filepath=f"/Users/jtremblay/code/mvs_objaverse/bowen.blend")

# raise()


# armature = Blender.Armature.Get(“Armature”)
# armobj = Blender.Object.Get(“Armature”)
# pose = armobj.getPose()

# for i in armature.bones.keys():
# pose.bones[i].loc = pose.bones[i].localMatrix.translationPart()
# pose.bones[i].quat = pose.bones[i].localMatrix.rotationPart().toQuat()
# pose.bones[i].size = pose.bones[i].localMatrix.scalePart()


for i in range(keys[-1]+1):
    render_single_image(
        frame_set = i,
        look_at_data=None,
        path = "/Users/jtremblay/code/mvs_objaverse/tmp/",
        resolution = RESOLUTION,
        )

    # for obj in bpy.data.objects:
    #     obj.select_set(False)
    # bpy.data.objects["Ch39"].select_set(True)

    bpy.ops.export_mesh.ply(filepath=f"/Users/jtremblay/code/mvs_objaverse/tmp/{str(i).zfill(3)}.ply",
        # use_materials=False,
        # axis_up='Z',
        )

    # raise()
# print(look_at_trans[0])




