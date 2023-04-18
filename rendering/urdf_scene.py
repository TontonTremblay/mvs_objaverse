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


def add_material(name,path):

    # Define the file paths for the texture maps
    ao_map_path = f"{path}/ao_map.jpg"
    color_map_path = f"{path}/color_map.jpg"
    displacement_map_path = f"{path}/displacement_map.jpg"
    normal_map_path = f"{path}/normal_map_opengl.jpg"
    roughness_map_path = f"{path}/roughness_map.jpg"
    metal_map_path = f"{path}/metalness_map.jpg"
    render_map_path = f"{path}/render_map.jpg"

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    bsdf = nodes.get("Principled BSDF") 

    node_color = nodes.new('ShaderNodeTexImage')
    node_color.image = bpy.data.images.load(color_map_path)
    node_tree.links.new(node_color.outputs["Color"], bsdf.inputs["Base Color"])

    normal_color = nodes.new('ShaderNodeTexImage')
    normal_color.image = bpy.data.images.load(normal_map_path)
    normal_color.image.colorspace_settings.name = 'Non-Color'

    normalmap = nodes.new(type="ShaderNodeNormalMap")

    node_tree.links.new(normal_color.outputs["Color"], normalmap.inputs["Color"])
    node_tree.links.new(normalmap.outputs["Normal"], bsdf.inputs["Normal"])

            
    rough_color = nodes.new('ShaderNodeTexImage')
    rough_color.image = bpy.data.images.load(roughness_map_path)
    rough_color.image.colorspace_settings.name = 'Non-Color'

    node_tree.links.new(rough_color.outputs["Color"], bsdf.inputs["Roughness"])

    if os.path.exists(metal_map_path):
        metal_color = nodes.new('ShaderNodeTexImage')
        metal_color.image = bpy.data.images.load(metal_map_path)
        metal_color.image.colorspace_settings.name = 'Non-Color'

        node_tree.links.new(rough_color.outputs["Color"], bsdf.inputs["Metallic"])


    return mat

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


##### LOAD THE URDF SCENE #####
urdf_content_path = "/Users/jtremblay/code/yourdfpy/robot/kitchen_urdf/"
kitchen = URDF.load(f"{urdf_content_path}/kitchen.urdf")

random.seed(100101)
np.random.seed(100101)

NB_FRAMES = 300

cfg_start = {}

for joint_name in kitchen.joint_names:
    j = kitchen.joint_map[joint_name]
    if not j.limit is None: 
        # print(j.limit)
        cfg_start[joint_name] = random.uniform(j.limit.lower,j.limit.upper)

#make the interpret
values_to_render = {}

for joint_name in cfg_start.keys():
    j = kitchen.joint_map[joint_name]
    nb_poses = 10
    x = np.linspace(0,1,nb_poses)
    y = [random.uniform(j.limit.lower,j.limit.upper) for i in range(nb_poses)]
    for iv in range(len(y)):
        if random.uniform(0,1)>0.7 :
            y[iv] = j.limit.lower
    inter = scipy.interpolate.interp1d(x,y,
        kind= random.choice(['linear', 'quadratic', 'cubic']),
    )
    # else:
    #     inter = scipy.interpolate.splrep(x,y,
    #         # kind= random.choice(['linear', 'quadratic', 'cubic']),
    #     )


    x_values = np.linspace(0,1,NB_FRAMES)
    
    values = inter(x_values)
    values[values < j.limit.lower] = j.limit.lower
    values[values > j.limit.upper] = j.limit.upper

    values_to_render[joint_name] = values

bpy.context.scene.frame_end=NB_FRAMES

kitchen.update_cfg(cfg_start)

# load in blender. 
link2blender = {}

metal = add_material('metal1','/Users/jtremblay/code/mvs_objaverse/assets/poly_textures/metal1/')
metal2 = add_material('metal2','/Users/jtremblay/code/mvs_objaverse/assets/poly_textures/metal2/')
wood = add_material('wood1','/Users/jtremblay/code/mvs_objaverse/assets/poly_textures/wood1/')
marble = add_material('marble1','/Users/jtremblay/code/mvs_objaverse/assets/poly_textures/marble1/')
plastic = add_material('plastic1','/Users/jtremblay/code/mvs_objaverse/assets/poly_textures/plastic1/')

def add_material(obj,material):
    if obj.data.materials:  
        # assign to 1st material slot
        obj.data.materials[0] = material
    else:
        # no slots
        obj.data.materials.append(material)
def add_light_under(obj,dist = 0.01,power=5): 
    light_data = bpy.data.lights.new(name=obj.name+"_light", type='POINT')
    light_data.energy = power
    light_object = bpy.data.objects.new(name=obj.name+"_light", object_data=light_data)
    light_object.location = (0, 0, -dist)
    bpy.context.collection.objects.link(light_object)
    light_object.parent = obj
    # raise()

for link in kitchen.link_map.keys():    
    link_name = link
    link = kitchen.link_map[link]

    for visual in link.visuals:
        if not 'range' in link_name:
            continue
        bpy.ops.object.select_all(action='DESELECT')
        if not visual.geometry.mesh is None:
            data_2_load = os.path.join(urdf_content_path,visual.geometry.mesh.filename)
            # print(data_2_load)
            bpy.ops.import_scene.obj(filepath=data_2_load)
            obj = bpy.context.selected_objects[0]
            obj.name = link_name
            bpy.context.view_layer.objects.active = obj
            # bpy.ops.wm.save_as_mainfile(filepath=f"/Users/jtremblay/code/mvs_objaverse/urdf.blend")
            # raise()
            bpy.ops.object.mode_set(mode='EDIT')
            # Select the geometry
            bpy.ops.mesh.select_all(action='SELECT')
            # Call the smart project operator
            bpy.ops.uv.smart_project()
            # Toggle out of Edit Mode
            bpy.ops.object.mode_set(mode='OBJECT')


            link2blender[link_name] = obj

        elif not visual.geometry.box is None:
            s = visual.geometry.box.size
            s = (s[0]/2,s[1]/2,s[2]/2)
            bpy.ops.mesh.primitive_cube_add()
            obj = bpy.context.active_object
            obj.scale = s
            bpy.ops.object.transform_apply(scale=True)
            obj.name = link_name
            link2blender[link_name] = obj
        elif not visual.geometry.cylinder is None:
            cyl = visual.geometry.cylinder
            bpy.ops.mesh.primitive_cylinder_add(radius=cyl.radius,depth=cyl.length)
            obj = bpy.context.active_object
            obj.name = link_name
            link2blender[link_name] = obj
        else:
            print(visual.geometry, 'not supported')

        if 'sink' in link_name:
            # print(link_name)
            if 'top' in link_name:
                add_material(obj,marble)
            elif 'handle' in link_name:
                add_material(obj,metal)
            elif 'cabinet' in link_name and not link_name.split("_")[-1] == 'sink':
                add_material(obj,wood)            
            elif 'door' in link_name:
                add_material(obj,wood)

            else:
                add_material(obj,metal)

        elif 'refrigerator' in link_name:
            add_material(obj,metal)
            if 'handle' in link_name:
                bpy.ops.object.shade_smooth()
                add_material(obj,metal2)
            if 'door' in link_name and 'holder' in link_name:
                bpy.ops.object.shade_smooth()
                add_material(obj,metal2)

            if 'top' in link_name:
                add_light_under(obj,dist=0.05)
            if 'separator' in link_name:
                add_light_under(obj,dist=0.05)
                add_material(obj,plastic)

            if 'shelf' in link_name and not 'door' in link_name:
                add_light_under(obj)
                add_material(obj,plastic)

        elif 'dishwasher' in link_name:
            if 'top' in link_name:
                add_material(obj,marble)
            else:
                add_material(obj,metal)
            if 'handle' in link_name:
                bpy.ops.object.shade_smooth()
                add_material(obj,metal2)
            if "surface" in link_name and link_name.split("_")[-1] == "0":
                # add a light under. 
                add_light_under(obj)

        elif 'range' in link_name:
            add_material(obj,metal)
            if 'handle' in link_name:
                bpy.ops.object.shade_smooth()
            if "hood" in link_name:
                # add a light under. 
                add_light_under(obj)
            if "surface" in link_name and link_name.split("_")[-1] == "0":
                # add a light under. 
                add_light_under(obj)
            if 'heater' in link_name: 
                add_material(obj,metal2)
            if 'handle' in link_name: 
                add_material(obj,metal2)

        elif 'cabinet' in link_name:
            if "wall" in link_name and 'surface' in link_name:
                # add a light under. 
                add_light_under(obj)

            if "handle" in link_name:
                add_material(obj,metal)

            elif 'surface' in link_name:
                add_material(obj,marble)
            elif 'top' in link_name:
                add_material(obj,marble)

            else:
                add_material(obj,wood)

        #### ADD ANNOTATION ####
        add_annotation(obj)

bpy.context.view_layer.update()


#### create the animation. 

k0 = list(values_to_render.keys())[0]
for i in range(len(values_to_render[k0])):
    cfg = {}
    for link in values_to_render.keys():
        cfg[link] = values_to_render[link][i]

    kitchen.update_cfg(cfg)

    for link in link2blender.keys():
        trans = kitchen.get_transform(link)
        obj = link2blender[link]
        
        obj.location.x = trans[0][-1]
        obj.location.y = trans[1][-1]
        obj.location.z = trans[2][-1]

        # print(trans)

        matrix = pyrr.Matrix44(trans).matrix33
        # print(matrix)
        # matrix = matrix * pyrr.Matrix33.from_x_rotation(-np.pi/2)
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion.x = matrix.quaternion.x
        obj.rotation_quaternion.y = matrix.quaternion.y
        obj.rotation_quaternion.z = matrix.quaternion.z
        obj.rotation_quaternion.w = matrix.quaternion.w

        obj.keyframe_insert(data_path='location', frame=i)
        obj.keyframe_insert(data_path='rotation_quaternion', frame=i)

# add a floor
bpy.ops.mesh.primitive_plane_add(size=2, 
    enter_editmode=False, 
    align='WORLD', 
    location=(0, 0, 0), 
    scale=(1, 1, 1)
)
ob = bpy.context.active_object
ob.name = "floor" 
# scaling_value = (5,3)
bpy.context.object.scale = (2.5,2,1)
bpy.context.object.location = (1.5,-1.5,0)

add_light_under(ob,-3,power=40)


####### make camera poses #########

positions_to_render = []
for ipos in range(NB_FRAMES):
    positions_to_render.append(
        [
            random.uniform(-1,1),
            random.uniform(-1,1),
            random.uniform(0,1),
        ]
    ) 

look_at_trans = []
global DATA_EXPORT

for pos in positions_to_render:

    at_name = list(DATA_EXPORT.keys())[np.random.randint(0,len(list(DATA_EXPORT.keys())))]
    pos_obj= bpy.data.objects[at_name].location
    look_at = [pos_obj[0],pos_obj[1],pos_obj[2]]

    look_at_trans.append({
        'at': look_at,
        'up': [0,0,1],
        'eye': [pos[0]+3,
                pos[1]-2,
                pos[2]+1]              
        }
    )

#### set up the camera 
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

cam = scene.objects['Camera']
cam.location = (0, 1.2, 0)  # radius equals to 1
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

bpy.context.scene.render.resolution_x = RESOLUTION
bpy.context.scene.render.resolution_y = RESOLUTION

make_segmentation_scene()

render_single_image(
    look_at_data=look_at_trans[0],
    path = "/Users/jtremblay/code/mvs_objaverse/tmp/",
    resolution = RESOLUTION,
    )
print(look_at_trans[0])
bpy.ops.wm.save_as_mainfile(filepath=f"/Users/jtremblay/code/mvs_objaverse/urdf.blend")




