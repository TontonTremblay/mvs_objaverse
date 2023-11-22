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
import argparse

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--blend_file', default=None,
    help='skip the scene set up')

argv = sys.argv[sys.argv.index("--") + 1:]
opt = parser.parse_args(argv)

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

s = 190191
random.seed(s)
np.random.seed(s)

NB_FRAMES = 300

cfg_start = {}
data_structure = {}

children_structure = {}
parent_structure = {}



for joint_name in kitchen.joint_names:
    j = kitchen.joint_map[joint_name]
    if not j.type == 'fixed':
        print(joint_name,j.type,j.origin)

    data_structure[joint_name] = {
        'parent':j.parent,
        'child':j.child
    }

    if j.parent in children_structure:
        children_structure[j.parent].append(j.child)
    else:
        children_structure[j.parent]=[j.child]
    parent_structure[j.child]=j.parent
    # if 'range' in j.child:
        # print(j.child,'is child to',j.parent)
    if not j.limit is None: 
        # print(j.limit)
        cfg_start[joint_name] = random.uniform(j.limit.lower,j.limit.upper)









# raise()
# print(data_structure)

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
    # print(link_name,link.parent,link.child)
    if len(link.visuals) == 0: 
        bpy.ops.object.empty_add(radius=0.05,location=(0,0,0))
        ob = bpy.context.object
        ob.name = link_name
        link2blender[link_name] = ob 
        add_annotation(ob,empty=True,link_name=link_name,data_parent=parent_structure)

    for visual in link.visuals:

        # TODO remove this> 

        # if not 'wall_cabinet_2' in link_name:
        #     continue


        # if not 'refrigerator' in link_name:
        #     continue

        # if not 'range' in link_name:
        #     continue

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
        add_annotation(obj,link_name= link_name,data_parent=parent_structure)

bpy.context.view_layer.update()

###### UPDATE THE KITCHEN to 0,0,0
cfg = {}
for link in values_to_render.keys():
    cfg[link] = values_to_render[link][0]

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

bpy.context.view_layer.update()


###### add the link positions. 

for joint_name in kitchen.joint_names:
    j = kitchen.joint_map[joint_name]
    if not j.type == 'fixed':
        if 'refrigerator' in joint_name:
            print(joint_name,j.type,j.origin)
        # get the obj in blender 
        ob_parent = bpy.data.objects[j.parent]
        # add empty 

        bpy.ops.object.empty_add(radius=0.05,
            location=(j.origin[0][-1],j.origin[1][-1],j.origin[2][-1])
        )
        ob_joint = bpy.context.object
        if j.type == 'prismatic':
            ob_joint.name = f"joint_min_pri_{j.name}"
        else:
            ob_joint.name = f"joint_min_rev_{j.name}"

        # ob_joint.parent = ob
        add_annotation(ob_joint,empty=True)
        # print(j.origin[:3])
        # t = np.array(j.origin)[:3,:3]

        # mat = mathutils.Matrix(t).to_euler()
        # print(mat)

        ob_joint.rotation_mode = 'AXIS_ANGLE'
        # ob_joint.rotation_euler[0] = j.axis[0]*(math.pi/180.0)
        # ob_joint.rotation_euler[1] = j.axis[1]*(math.pi/180.0)
        # ob_joint.rotation_euler[2] = j.axis[2]*(math.pi/180.0) 
        
        ob_joint.rotation_axis_angle[0] = j.axis[0]
        ob_joint.rotation_axis_angle[1] = j.axis[1]
        ob_joint.rotation_axis_angle[2] = j.axis[2]
        # ob_joint.rotation_axis_angle[3] = j.limit.lower
        # print(ob_joint.rotation_axis_angle[3]) 


        # add the limit
        if j.type == 'prismatic':
            if j.axis[0] > 0 or j.axis[0] < 0 : 
                bpy.ops.object.empty_add(radius=0.05,
                    location=(j.axis[0] * j.limit.upper,0,0)
                )
            elif j.axis[1] > 0 or j.axis[1] < 0 : 
                bpy.ops.object.empty_add(radius=0.05,
                    location=(0,j.axis[1] * j.limit.upper,0)
                )
            elif j.axis[2] > 0 or j.axis[2] < 0 : 
                bpy.ops.object.empty_add(radius=0.05,
                    location=(0,0,j.axis[2] * j.limit.upper)
                )
            ob_joint.parent = ob_parent

            ob_joint_max = bpy.context.object
            ob_joint_max.name = f"joint_max_pri_{j.name}"
            ob_joint_max.parent = ob_joint
            add_annotation(ob_joint_max,empty=True)

        # find width and/or height
        if j.type == 'revolute':
            ob_child = bpy.data.objects[j.child]
            #find size
            print(ob_child.name)
            to_check_size =[]
            for ob in bpy.data.objects:
                if ob_child.name in ob.name and ob.type=="MESH":
                    to_check_size.append(ob)
                    print(' ',ob.name)
            def get_dim(mesh_objs):
                corners = []
                for ob in mesh_objs:
                    print(ob.name,ob.matrix_world)
                    if not ob.type == "MESH":
                        continue
                    ob.select_set(True)
                    bpy.context.view_layer.objects.active = ob
                    bpy.context.view_layer.update()

                    bbox_corners = [ob.matrix_world @ Vector(corner)  for corner in ob.bound_box]
                    # bbox_corners = [Vector(corner)  for corner in ob.bound_box]
                    
                    for corn in bbox_corners:
                        corners.append([corn.x,corn.y,corn.z])

                corners = np.array(corners)

                # bpy.ops.mesh.primitive_ico_sphere_add(scale=(0.1,0.1,0.1),location=(np.min(corners[:,0]),np.min(corners[:,1]),np.min(corners[:,2])))
                # bpy.ops.mesh.primitive_ico_sphere_add(scale=(0.1,0.1,0.1),location=(np.max(corners[:,0]),np.max(corners[:,1]),np.max(corners[:,2])))
                mina = [np.min(corners[:,0]),np.min(corners[:,1]),np.min(corners[:,2])]
                maxb = [np.max(corners[:,0]),np.max(corners[:,1]),np.max(corners[:,2])]
                # center_point = Vector(((minA.x + maxB.x)/2, (minA.y + maxB.y)/2, (minA.z + maxB.z)/2))
                center_point = Vector(((mina[0] + maxb[0])/2, (mina[1] + maxb[1])/2, (mina[2] + maxb[2])/2))
                # bpy.ops.mesh.primitive_ico_sphere_add(scale=(0.1,0.1,0.1),location=(center_point))
                # center_point = 
                dimensions =  Vector((maxb[0] - mina[0], maxb[1] - mina[1], maxb[2] - mina[2]))
                return {"dim":dimensions,"corners":corners,'minmax':[mina,maxb]}
            if len(to_check_size)==0:
                x,y,z = 1,1,1
                r = None
            else:
                r = get_dim(to_check_size)
                x,y,z = r['dim']
            # print(j.name,x,y,z)

            # dot product
            def project_on_line(point, line_point, line_axis):
                # convert inputs to numpy arrays for ease of computation
                point = np.array(point)
                line_point = np.array(line_point)
                line_axis = np.array(line_axis)

                # compute the projection of the point onto the line
                projection = line_point + np.dot(point - line_point, line_axis) * line_axis

                # convert the projection back to a tuple for ease of use
                return tuple(projection)

            if not r is None:
                all_ = []
                for ip, p in enumerate(r['corners']):
                    all_.append(project_on_line(p,
                        ob_parent.matrix_world @ Vector([j.origin[0][-1],j.origin[1][-1],j.origin[2][-1]]),
                        j.axis))
                    print(p, "->",all_[-1])

                ps = np.array(all_)
                # print(p, "->",ps)
            else:
                ps = np.array([[1,1,1]])

            ob_joint.location.x = np.min(ps[:,0])
            ob_joint.location.y = np.min(ps[:,1])
            ob_joint.location.z = np.min(ps[:,2])

            bpy.ops.object.empty_add(radius=0.05,
                location=(
                    np.max(ps[:,0]),
                    np.max(ps[:,1]),
                    np.max(ps[:,2])
                )
            )

            ob_joint_max = bpy.context.object
            ob_joint_max.name = f"joint_max_rev_{j.name}"
            # ob_joint_max.parent = ob_joint
            add_annotation(ob_joint_max,empty=True)


# raise()
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

    while True:
        at_name = list(DATA_EXPORT.keys())[np.random.randint(0,len(list(DATA_EXPORT.keys())))]
        if bpy.data.objects[at_name].type == 'MESH':
            break
    pos_obj= bpy.data.objects[at_name].location
    look_at = [pos_obj[0],pos_obj[1],pos_obj[2]]

    look_at_trans.append({
        'at': look_at,
        'up': [0,0,1],
        'eye': [pos[0]+2,
                pos[1]-5,
                pos[2]+0.5]              
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
# print(look_at_trans[0])
bpy.ops.wm.save_as_mainfile(filepath=f"/Users/jtremblay/code/mvs_objaverse/urdf.blend")




