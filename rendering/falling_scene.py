import argparse, sys, os, math, re
import bpy
from mathutils import Vector, Matrix
import mathutils
import numpy as np
import json 
import random 
import glob 

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--views', type=int, default=24,
    help='number of views to be rendered')
parser.add_argument(
    '--folder_assets', type=str,
    help='Path to the obj file to be rendered.')
parser.add_argument(
    '--output_folder', type=str, default='/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/shapenet/renders/',
    help='The path the output will be dumped to.')
parser.add_argument(
    '--format', type=str, default='PNG',
    help='Format of files generated. Either PNG or OPEN_EXR')

parser.add_argument(
    '--input_model', type=str, default='glb',
    help='glb is the format for objaverse, but we can use [obj,ply,glb]')
parser.add_argument(
    '--outf_name', type=str, default=None,
    help='folder to put things in')

parser.add_argument(
    '--use_model_identifier', action='store_true',
    help='add the name of the folder to the end of the thing.')

parser.add_argument(
    '--resolution', type=int, default=256,
    help='Resolution of the images.')
parser.add_argument(
    '--engine', type=str, default='CYCLES',
    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

parser.add_argument(
    '--asset_textures', type=str, default='/home/jtremblay/code/visii_mvs/cco_textures/',
    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

parser.add_argument(
    '--assets_hdri', type=str, default='/home/jtremblay/code/visii_mvs/dome_hdri_haven/',
    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
parser.add_argument(
    '--save_tmp_blend', type=str, default='',
    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')


argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def enable_cuda_devices():
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()

    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            print("Compute device selected: {0}".format(compute_device_type))
            break
        except TypeError:
            pass

    # Any CUDA/OPENCL devices?
    acceleratedTypes = ['CUDA', 'OPENCL']
    accelerated = any(device.type in acceleratedTypes for device in cprefs.devices)
    print('Accelerated render = {0}'.format(accelerated))

    # If we have CUDA/OPENCL devices, enable only them, otherwise enable
    # all devices (assumed to be CPU)
    print(cprefs.devices)
    for device in cprefs.devices:
        device.use = not accelerated or device.type in acceleratedTypes
        print('Device enabled ({type}) = {enabled}'.format(type=device.type, enabled=device.use))

    return accelerated



def add_planes():
    world = bpy.data.worlds['World']
    world.use_nodes = True

    # add a planes
    ob = bpy.ops.mesh.primitive_plane_add(size=2, 
        enter_editmode=False, 
        align='WORLD', 
        location=(0, 0, 0), 
        scale=(1, 1, 1)
    )

    bpy.context.object.scale = (15,15,1)
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.collision_shape = 'MESH'

    # add the texture 
    texture = glob.glob(f'{args.asset_textures}*/')
    texture_random_selection = texture[random.randint(0,len(texture)-1)]

    files = glob.glob(texture_random_selection+'/*.jpg')+glob.glob(texture_random_selection+'/*.png')
    ob = bpy.context.active_object
    mat = bpy.data.materials.new(name="floor")

    # Assign it to object
    if ob.data.materials:
        # assign to 1st material slot
        ob.data.materials[0] = mat
    else:
        # no slots
        ob.data.materials.append(mat)
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    bsdf = nodes.get("Principled BSDF") 
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    node_tree.links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    mapping.inputs['Scale'].default_value[0] = 1
    mapping.inputs['Scale'].default_value[1] = 1
    
    for file in files:
        if 'color' in file.lower():
            node_color = nodes.new('ShaderNodeTexImage')
            node_color.image = bpy.data.images.load(file)
            node_tree.links.new(mapping.outputs["Vector"], node_color.inputs["Vector"])
            node_tree.links.new(node_color.outputs["Color"], bsdf.inputs["Base Color"])

        if 'normal' in file.lower():
            normal_color = nodes.new('ShaderNodeTexImage')
            normal_color.image = bpy.data.images.load(file)
            normal_color.image.colorspace_settings.name = 'Non-Color'

            normalmap = nodes.new(type="ShaderNodeNormalMap")

            node_tree.links.new(normal_color.outputs["Color"], normalmap.inputs["Color"])
            node_tree.links.new(normalmap.outputs["Normal"], bsdf.inputs["Normal"])


            
        if 'rough' in file.lower():
            rough_color = nodes.new('ShaderNodeTexImage')
            rough_color.image = bpy.data.images.load(file)
            rough_color.image.colorspace_settings.name = 'Non-Color'

            node_tree.links.new(mapping.outputs["Vector"], rough_color.inputs["Vector"])
            node_tree.links.new(rough_color.outputs["Color"], bsdf.inputs["Roughness"])


        if 'metal' in file.lower():
            metal_color = nodes.new('ShaderNodeTexImage')
            metal_color.image = bpy.data.images.load(file)
            metal_color.image.colorspace_settings.name = 'Non-Color'

            node_tree.links.new(mapping.outputs["Vector"], metal_color.inputs["Vector"])
            node_tree.links.new(metal_color.outputs["Color"], bsdf.inputs["Metallic"])


    normal_map = world.node_tree.nodes.new('ShaderNodeNormalMap')
    print(bsdf.inputs.keys())

    # raise()
    bpy.ops.object.mode_set(mode='OBJECT')

    ob = bpy.ops.mesh.primitive_plane_add(size=2, 
        enter_editmode=False, 
        align='WORLD', 
        location=(0, -15, 10), 
        scale=(1, 1, 1)
    )

    bpy.context.object.scale = (15,15,1)
    bpy.context.object.rotation_mode = 'XYZ'
    bpy.context.object.rotation_euler = (np.pi/2,0,0)

    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.context.object.hide_render = True
    bpy.context.object.hide_viewport = True


    ob = bpy.ops.mesh.primitive_plane_add(size=2, 
        enter_editmode=False, 
        align='WORLD', 
        location=(0, 15, 10), 
        scale=(1, 1, 1)
    )

    bpy.context.object.scale = (15,15,1)
    bpy.context.object.rotation_mode = 'XYZ'
    bpy.context.object.rotation_euler = (np.pi/2,0,0)

    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.context.object.hide_render = True
    bpy.context.object.hide_viewport = True


    ob = bpy.ops.mesh.primitive_plane_add(size=2, 
        enter_editmode=False, 
        align='WORLD', 
        location=(-15, 0, 10), 
        scale=(1, 1, 1)
    )

    bpy.context.object.scale = (15,15,1)
    bpy.context.object.rotation_mode = 'XYZ'
    bpy.context.object.rotation_euler = (0,np.pi/2,0)

    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.context.object.hide_render = True
    bpy.context.object.hide_viewport = True

    ob = bpy.ops.mesh.primitive_plane_add(size=2, 
        enter_editmode=False, 
        align='WORLD', 
        location=(15, 0, 10), 
        scale=(1, 1, 1)
    )

    bpy.context.object.scale = (15,15,1)
    bpy.context.object.rotation_mode = 'XYZ'
    bpy.context.object.rotation_euler = (0,np.pi/2,0)

    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.context.object.hide_render = True
    bpy.context.object.hide_viewport = True


def random_sample_sphere(
        elevation_range = [0,180],
        tetha_range = [0,360],
        nb_frames = 10,
    ):
    to_return = []
    outside = True
    max_radius = 1.00001
    min_radius = 0.99999
    min_max_x = [0,0]
    min_max_y = [0,0]
    min_max_z = [0,0]
    for i_degree in range(tetha_range[0],tetha_range[1],1):
        v = np.cos(np.deg2rad(i_degree))
        if v < min_max_x[0]:
            min_max_x[0] = v
        if v > min_max_x[1]:
            min_max_x[1] = v

    for i_degree in range(tetha_range[0],tetha_range[1],1):
        v = np.sin(np.deg2rad(i_degree))
        if v < min_max_y[0]:
            min_max_y[0] = v
        if v > min_max_y[1]:
            min_max_y[1] = v

    for i_degree in range(elevation_range[0],elevation_range[1],1):
        v = np.cos(np.deg2rad(i_degree))
        if v < min_max_z[0]:
            min_max_z[0] = v
        if v > min_max_z[1]:
            min_max_z[1] = v

    for i in range(nb_frames):
        outside = True
        while outside:

            x = random.uniform(min_max_x[0], min_max_x[1])
            y = random.uniform(min_max_y[0], min_max_y[1])
            z = random.uniform(min_max_z[0], min_max_z[1])

            if  (x**2 + y**2 + z**2) * max_radius < max_radius + 0.0001 \
            and (x**2 + y**2 + z**2) * max_radius > min_radius:
                outside = False
        to_return.append([x,y,z])
    return to_return

class Bezier():
    def TwoPoints(t, P1, P2):
        """
        Returns a point between P1 and P2, parametised by t.
        INPUTS:
            t     float/int; a parameterisation.
            P1    numpy array; a point.
            P2    numpy array; a point.
        OUTPUTS:
            Q1    numpy array; a point.
        """

        if not isinstance(P1, np.ndarray) or not isinstance(P2, np.ndarray):
            raise TypeError('Points must be an instance of the numpy.ndarray!')
        if not isinstance(t, (int, float)):
            raise TypeError('Parameter t must be an int or float!')

        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def Points(t, points):
        """
        Returns a list of points interpolated by the Bezier process
        INPUTS:
            t            float/int; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoints    list of numpy arrays; points.
        """
        newpoints = []
        #print("points =", points, "\n")
        for i1 in range(0, len(points) - 1):
            #print("i1 =", i1)
            #print("points[i1] =", points[i1])

            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
            #print("newpoints  =", newpoints, "\n")
        return newpoints

    def Point(t, points):
        """
        Returns a point interpolated by the Bezier process
        INPUTS:
            t            float/int; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoint     numpy array; a point.
        """
        newpoints = points
        #print("newpoints = ", newpoints)
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
            #print("newpoints in loop = ", newpoints)

        #print("newpoints = ", newpoints)
        #print("newpoints[0] = ", newpoints[0])
        return newpoints[0]

    def Curve(t_values, points):
        """
        Returns a point interpolated by the Bezier process
        INPUTS:
            t_values     list of floats/ints; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            curve        list of numpy arrays; points.
        """

        if not hasattr(t_values, '__iter__'):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if len(t_values) < 1:
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if not isinstance(t_values[0], (int, float)):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")

        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            #print("curve                  \n", curve)
            #print("Bezier.Point(t, points) \n", Bezier.Point(t, points))

            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)

            #print("curve after            \n", curve, "\n--- --- --- --- --- --- ")
        curve = np.delete(curve, 0, 0)
        #print("curve final            \n", curve, "\n--- --- --- --- --- --- ")
        return curve

# Function taken from https://github.com/zhenpeiyang/HM3D-ABO/blob/master/my_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = mathutils.Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

# Function taken from https://github.com/zhenpeiyang/HM3D-ABO/blob/master/my_blender.py
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = mathutils.Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

# function taken from https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
def look_at(obj, target, roll=0):
    """
    Rotate obj to look at target

    :arg obj: the object to be rotated. Usually the camera
    :arg target: the location (3-tuple or Vector) to be looked at
    :arg roll: The angle of rotation about the axis from obj to target in radians. 

    Based on: https://blender.stackexchange.com/a/5220/12947 (ideasman42)      
    """
    if not isinstance(target, mathutils.Vector):
        target = mathutils.Vector(target)
    loc = obj.location
    # direction points from the object to the target
    direction = target - loc
    tracker, rotator = (('-Z', 'Y'),'Z') if obj.type=='CAMERA' else (('X', 'Z'),'Y') #because new cameras points down(-Z), usually meshes point (-Y)
    quat = direction.to_track_quat(*tracker)
    
    # /usr/share/blender/scripts/addons/add_advanced_objects_menu/arrange_on_curve.py
    quat = quat.to_matrix().to_4x4()
    rollMatrix = mathutils.Matrix.Rotation(roll, 4, rotator)

    # remember the current location, since assigning to obj.matrix_world changes it
    loc = loc.to_tuple()
    #obj.matrix_world = quat * rollMatrix
    # in blender 2.8 and above @ is used to multiply matrices
    # using * still works but results in unexpected behaviour!
    obj.matrix_world = quat @ rollMatrix
    obj.location = loc

def export_to_ndds_file(
    filename = "tmp.json", #this has to include path as well
    obj_names = [], # this is a list of ids to load and export
    height = 500, 
    width = 500,
    camera_name = 'my_camera',
    cuboids = None,
    camera_struct = None,
    segmentation_mask = None,
    visibility_percentage = False, 
    scene_aabb = None, #min,max,center 
    ):
    # To do export things in the camera frame, e.g., pose and quaternion

    import simplejson as json

    # assume we only use the view camera
    cam_matrix = visii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
    
    # print("get_world_to_local_matrix")
    # print(cam_matrix)
    # print('look_at')
    # print(camera_struct)
    # print('position')
    # print(visii.entity.get(camera_name).get_transform().get_position())
    # # raise()
    cam_matrix_export = []
    for row in cam_matrix:
        cam_matrix_export.append([row[0],row[1],row[2],row[3]])
    
    cam_world_location = visii.entity.get(camera_name).get_transform().get_position()
    cam_world_quaternion = visii.entity.get(camera_name).get_transform().get_rotation()
    # cam_world_quaternion = visii.quat_cast(cam_matrix)

    cam_intrinsics = visii.entity.get(camera_name).get_camera().get_intrinsic_matrix(width, height)

    if camera_struct is None:
        camera_struct = {
            'at': [0,0,0,],
            'eye': [0,0,0,],
            'up': [0,0,0,]
        }
    cam2wold = visii.entity.get(camera_name).get_transform().get_local_to_world_matrix()
    cam2wold_export = []
    for row in cam2wold:
        cam2wold_export.append([row[0],row[1],row[2],row[3]])
    if scene_aabb is None:
        scene_aabb = [
            [
                visii.get_scene_min_aabb_corner()[0],
                visii.get_scene_min_aabb_corner()[1],
                visii.get_scene_min_aabb_corner()[2],
            ],
            [
                visii.get_scene_max_aabb_corner()[0],
                visii.get_scene_max_aabb_corner()[1],
                visii.get_scene_max_aabb_corner()[2],
            ],
            [
                visii.get_scene_aabb_center()[0],
                visii.get_scene_aabb_center()[1],
                visii.get_scene_aabb_center()[2],
            ]
        ]
    dict_out = {
                "camera_data" : {
                    "width" : width,
                    'height' : height,
                    'camera_look_at':
                    {
                        'at': [
                            camera_struct['at'][0],
                            camera_struct['at'][1],
                            camera_struct['at'][2],
                        ],
                        'eye': [
                            camera_struct['eye'][0],
                            camera_struct['eye'][1],
                            camera_struct['eye'][2],
                        ],
                        'up': [
                            camera_struct['up'][0],
                            camera_struct['up'][1],
                            camera_struct['up'][2],
                        ]
                    },
                    'camera_view_matrix':cam_matrix_export,
                    'cam2world':cam2wold_export,
                    'location_world':
                    [
                        cam_world_location[0],
                        cam_world_location[1],
                        cam_world_location[2],
                    ],
                    'quaternion_world_xyzw':[
                        cam_world_quaternion[0],
                        cam_world_quaternion[1],
                        cam_world_quaternion[2],
                        cam_world_quaternion[3],
                    ],
                    'intrinsics':{
                        'fx':cam_intrinsics[0][0],
                        'fy':cam_intrinsics[1][1],
                        'cx':cam_intrinsics[2][0],
                        'cy':cam_intrinsics[2][1]
                    },
                    'scene_min_3d_box':scene_aabb[0],
                    'scene_max_3d_box':scene_aabb[1],
                    'scene_center_3d_box':scene_aabb[2],
                }, 
                "objects" : []
            }

    # Segmentation id to export
    id_keys_map = visii.entity.get_name_to_id_map()

    for obj_name in obj_names: 

        projected_keypoints, _ = get_cuboid_image_space(obj_name, camera_name=camera_name)

        # put them in the image space. 
        for i_p, p in enumerate(projected_keypoints):
            projected_keypoints[i_p] = [p[0]*width, p[1]*height]

        # Get the location and rotation of the object in the camera frame 


        trans = visii.transform.get(obj_name)
        if trans is None: 
            trans = visii.entity.get(obj_name).get_transform()
            
        quaternion_xyzw = visii.inverse(cam_world_quaternion) * trans.get_rotation()

        object_world = visii.vec4(
            trans.get_position()[0],
            trans.get_position()[1],
            trans.get_position()[2],
            1
        ) 
        pos_camera_frame = cam_matrix * object_world
 
        if not cuboids is None and obj_name in cuboids:
            cuboid = cuboids[obj_name]
        else:
            cuboid = None

        #check if the object is visible
        visibility = -1
        bounding_box = [-1,-1,-1,-1]

        if segmentation_mask is None:
            segmentation_mask = visii.render_data(
                width=int(width), 
                height=int(height), 
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options="entity_id",
            )
            segmentation_mask = np.array(segmentation_mask).reshape(width,height,4)[:,:,0]
            
        if visibility_percentage == True and int(id_keys_map [obj_name]) in np.unique(segmentation_mask.astype(int)): 
            transforms_to_keep = {}
            
            for name in id_keys_map.keys():
                if 'camera' in name.lower() or obj_name in name:
                    continue
                trans_to_keep = visii.entity.get(name).get_transform()
                transforms_to_keep[name]=trans_to_keep
                visii.entity.get(name).clear_transform()

            # Percentatge visibility through full segmentation mask. 
            segmentation_unique_mask = visii.render_data(
                width=int(width), 
                height=int(height), 
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options="entity_id",
            )

            segmentation_unique_mask = np.array(segmentation_unique_mask).reshape(width,height,4)[:,:,0]

            values_segmentation = np.where(segmentation_mask == int(id_keys_map[obj_name]))[0]
            values_segmentation_full = np.where(segmentation_unique_mask == int(id_keys_map[obj_name]))[0]
            visibility = len(values_segmentation)/float(len(values_segmentation_full))
            
            # bounding box calculation

            # set back the objects from remove
            for entity_name in transforms_to_keep.keys():
                visii.entity.get(entity_name).set_transform(transforms_to_keep[entity_name])
        else:
            # print(np.unique(segmentation_mask.astype(int)))
            # print(np.isin(np.unique(segmentation_mask).astype(int),
            #         [int(name_to_id[obj_name])]))
            try:
                if int(id_keys_map[obj_name]) in np.unique(segmentation_mask.astype(int)): 
                    #
                    visibility = 1
                    y,x = np.where(segmentation_mask == int(id_keys_map[obj_name]))
                    bounding_box = [int(min(x)),int(max(x)),height-int(max(y)),height-int(min(y))]
                else:
                    visibility = 0
            except:
                visibility= -1

        tran_matrix = trans.get_local_to_world_matrix()
    
        trans_matrix_export = []
        for row in tran_matrix:
            trans_matrix_export.append([row[0],row[1],row[2],row[3]])

        # Final export
        dict_out['objects'].append({
            # 'class':obj_name.split('_')[1],
            'name':obj_name,
            'provenance':'visii',
            # TODO check the location
            'location': [
                pos_camera_frame[0],
                pos_camera_frame[1],
                pos_camera_frame[2]
            ],
            'location_world': [
                trans.get_position()[0],
                trans.get_position()[1],
                trans.get_position()[2]
            ],
            'quaternion_xyzw':[
                quaternion_xyzw[0],
                quaternion_xyzw[1],
                quaternion_xyzw[2],
                quaternion_xyzw[3],
            ],
            'quaternion_xyzw_world':[
                trans.get_rotation()[0],
                trans.get_rotation()[1],
                trans.get_rotation()[2],
                trans.get_rotation()[3]
            ],
            'local_to_world_matrix':trans_matrix_export,
            'projected_cuboid':projected_keypoints,
            'local_cuboid': cuboid,
            'visibility':visibility,
            'bounding_box_minx_maxx_miny_maxy':bounding_box,
        })
        try:
            dict_out['objects'][-1]['segmentation_id']=id_keys_map[obj_name]
        except:
            dict_out['objects'][-1]['segmentation_id']=-1

        try:
            dict_out['objects'][-1]['mat_metallic']=visii.entity.get(obj_name).get_material().get_metallic()
            dict_out['objects'][-1]['mat_roughness']=visii.entity.get(obj_name).get_material().get_roughness()
            dict_out['objects'][-1]['mat_transmission']=visii.entity.get(obj_name).get_material().get_transmission()
            dict_out['objects'][-1]['mat_sheen']=visii.entity.get(obj_name).get_material().get_sheen()
            dict_out['objects'][-1]['mat_clearcoat']=visii.entity.get(obj_name).get_material().get_clearcoat()
            dict_out['objects'][-1]['mat_specular']=visii.entity.get(obj_name).get_material().get_specular()
            dict_out['objects'][-1]['mat_anisotropic']=visii.entity.get(obj_name).get_material().get_anisotropic()            
        except:
            pass            
    with open(filename, 'w+') as fp:
        json.dump(dict_out, fp, indent=4, sort_keys=True)
    # return bounding_box


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = args.engine
render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
render.image_settings.file_format = args.format  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
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



NB_OBJECTS_LOADED = 5
NB_OBJECTS_LOADED_OTHERS = 0

enable_cuda_devices()
# context.active_object.select_set(True)
# bpy.ops.object.delete()
# for ob in bpy.context.scene.objects:
#     ob.select_set(True)
bpy.ops.object.delete()
# bpy.ops.objects['Light'].delete()
bpy.data.objects['Light'].select_set(True)
bpy.ops.object.delete()

# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')

bpy.ops.rigidbody.world_add()
bpy.context.scene.rigidbody_world.enabled = True

if args.input_model == "glb":
    assets_content = glob.glob(f"{args.folder_assets}/*.{args.input_model}")

    for i in range(NB_OBJECTS_LOADED): 
        to_load = assets_content[random.randint(0,len(assets_content)-1)]
        print(to_load)
        imported_object = bpy.ops.import_scene.gltf(filepath=to_load)

        for ob in bpy.context.selected_objects:
            if ob.type == 'MESH':
                break
        bpy.ops.rigidbody.object_add({'object': ob})
        ob.rigid_body.collision_shape = 'BOX'


# load some distractors 

obj_to_export = []

bpy.ops.object.select_all(action='DESELECT')
assets_content = glob.glob("/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/google_scanned/*/")
for i in range(NB_OBJECTS_LOADED_OTHERS): 
    to_load = assets_content[random.randint(0,len(assets_content)-1)]
    imported_object = bpy.ops.import_scene.obj(filepath=f"{to_load}/meshes/model.obj")
    # bpy.ops.transform.resize(value=(7, 7, 7))


# "meshes/model.obj"
# spread the hammers
# bpy.context.scene = 'PHYSICS'



for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        ob.rotation_mode = 'XYZ'
        ob.rotation_euler = (random.randint(-100,100),random.randint(-100,100),random.randint(-100,100))
        ob.location = (random.randrange(-5,5),random.randrange(-5,5),random.randrange(4,8))
        # ob.select_set(True)
        # bpy.data.scenes['Scene'].rigidbody_world.collection.objects.link(ob)
        # bpy.ops.rigidbody.object_add()

        if 'model' in ob.name:
            s = random.randint(6,10)
            ob.scale = (s,s,s)
            bpy.ops.rigidbody.object_add({'object': ob})
            ob.rigid_body.collision_shape = 'BOX'

            # bpy.ops.rigidbody.objects_add()
            # bpy.context.object.rigid_body.collision_shape = 'BOX'
        else:
            obj_to_export.append(ob.name)

# add some boxes


add_planes()


# HDR LIGHT
world = bpy.data.worlds['World']
world.use_nodes = True
bg = world.node_tree.nodes['Background']

node_environment = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
# Load and assign the image to the node property
skyboxes = glob.glob(f'{args.assets_hdri}/*.hdr')
skybox_random_selection = skyboxes[random.randint(0,len(skyboxes)-1)]

node_environment.image = bpy.data.images.load(skybox_random_selection) # Relative path
# node_environment.location = -300,0

world.node_tree.links.new(node_environment.outputs["Color"], bg.inputs["Color"])

# for scene in bpy.ops.scenes.keys():
    # print(scene)
# print()

# raise()



# simulate to frame 10. 
# bpy.ops.screen.animation_play()

# Run the physics simulation
scene = bpy.context.scene
scene.frame_set(0)
bpy.context.view_layer.update()

# for jj in range(250):
bpy.context.scene.rigidbody_world.substeps_per_frame = 10
bpy.context.scene.rigidbody_world.solver_iterations = 10
bpy.context.scene.rigidbody_world.enabled = True

# Run simulation
point_cache = bpy.context.scene.rigidbody_world.point_cache
point_cache.frame_start = 1

for i in range(200):
    point_cache.frame_end = i
    bpy.ops.ptcache.bake({"point_cache": point_cache}, bake=True)
    scene.frame_set(i)
    bpy.context.view_layer.update()
# scene.frame_set(100)

# bpy.data.scenes['Scene'].frame_set(100)



































# generate camera poses 
cfg = {}
cfg['camera_nb_anchor'] = 40
cfg['camera_elevation_range'] = [0,360]
cfg['camera_theta_range'] = [30,50]
cfg['to_add_position'] = [8,8,5]
cfg['camera_fixed_distance_factor'] = 5
cfg['camera_nb_frames'] = 300
cfg = dotdict(cfg)

anchor_points = random_sample_sphere(
                        nb_frames = cfg.camera_nb_anchor,
                        elevation_range = cfg.camera_elevation_range,
                        tetha_range = cfg.camera_theta_range
                    )
anchor_points[-1] = anchor_points[0]

t_points = np.arange(0, 1, 1/cfg.camera_nb_frames) #................................. Creates an iterable list from 0 to 1.
anchor_points = np.array(anchor_points)
positions_to_render = Bezier.Curve(t_points, anchor_points) #......................... Returns an array of coordinates.

at_all = []
for i_at in range(len(anchor_points)):
    # if random.random() < 0.5:
    #     at = entity_list[np.random.randint(
    #         0,
    #         len(entity_list))
    #     ].get_transform().get_position()
    #     at = [at[0],at[1],at[2]]
    at = [0,0,0]
    at_all.append(at)
at_all[-1] = at_all[0]    
at_all = np.array(at_all)
at_all = Bezier.Curve(t_points, at_all) #......................... Returns an array of coordinates.


# generate the camera poses 
look_at_trans = []
to_add = [0,0,0]
if "to_add_position" in cfg:
    to_add = cfg.to_add_position
for i_pos, pos in enumerate(positions_to_render):
    look_at_trans.append({
        'at': at_all[i_pos],
        'up': [0,0,1],
        'eye': [pos[0]*float(cfg.camera_fixed_distance_factor)+to_add[0],
                pos[1]*float(cfg.camera_fixed_distance_factor)+to_add[1],
                pos[2]*float(cfg.camera_fixed_distance_factor)+to_add[2]]              
        }
    )
# print(positions_to_render)

# Place camera




cam = scene.objects['Camera']
cam.location = (0, 1.2, 0)  # radius equals to 1
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty


bpy.context.scene.render.resolution_x = args.resolution
bpy.context.scene.render.resolution_y = args.resolution


K = get_calibration_matrix_K_from_blender(bpy.data.cameras[0])

if args.use_model_identifier:
    model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
    path = os.path.join(os.path.abspath(args.output_folder), model_identifier)
else:
    path = os.path.abspath(args.output_folder)

if not args.outf_name is None:
    path = os.path.join(os.path.abspath(args.output_folder), args.outf_name)

to_export = {
    # 'fx': K[0][0],
    # 'fy': K[1][1],
    # 'cx': K[0][-1],
    # 'cy': K[1][-1],
    'camera_angle_x': bpy.data.cameras[0].angle_x,
    "aabb": [
        [
            0.5,
            0.5,
            0.5
        ],
        [
            -0.5,
            -0.5,
            -0.5
        ]
    ],
}




##### CREATE A new scene for segmentation rendering 
result = bpy.ops.scene.new(type='FULL_COPY')
bpy.context.scene.name = "segmentation"


# lets update all the materials of the objects to emmissive
to_change = []
for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        to_change.append(ob)

def _colorize_object(obj: bpy.types.Object, color: mathutils.Vector, use_alpha_channel: bool):
    """ Adjusts the materials of the given object, s.t. they are ready for rendering the seg map.
    This is done by replacing all nodes just with an emission node, which emits the color corresponding to the
    category of the object.
    :param obj: The object to use.
    :param color: RGB array of a color in the range of [0, self.render_colorspace_size_per_dimension].
    :param use_alpha_channel: If true, the alpha channel stored in .png textures is used.
    """
    # Create new material emitting the given color
    new_mat = bpy.data.materials.new(name="segmentation")
    new_mat.use_nodes = True
    # sampling as light,conserves memory, by not keeping a reference to it for multiple importance sampling.
    # This shouldn't change the results because with an emission of 1 the colorized objects aren't emitting light.
    # Also, BlenderProc's segmap render settings are configured so that there is only a single sample to distribute,
    # multiple importance shouldn't affect the noise of the render anyway.
    # This fixes issue #530
    new_mat.cycles.sample_as_light = False
    nodes = new_mat.node_tree.nodes
    links = new_mat.node_tree.links
    emission_node = nodes.new(type='ShaderNodeEmission')
    # output = nodes.new('OutputMaterial')
    output = nodes.get("Material Output")

    emission_node.inputs['Color'].default_value[:3] = color
    links.new(emission_node.outputs['Emission'], output.inputs['Surface'])

    # Set material to be used for coloring all faces of the given object
    if len(obj.material_slots) > 0:
        for i, material_slot in enumerate(obj.material_slots):
            if use_alpha_channel:
                obj.data.materials[i] = MaterialLoaderUtility.add_alpha_texture_node(material_slot.material,
                                                                                     new_mat)
            else:
                obj.data.materials[i] = new_mat
    else:
        obj.data.materials.append(new_mat)


import colorsys
for ob in to_change:

    c = colorsys.hsv_to_rgb(
        random.randrange(0,255)/255, 
        random.randrange(200,255)/255, 
        random.randrange(200,255)/255
        )
    print(c)
    # c = [c[0]/255.0,c[1]/255.0,c[2]/255.0]
    # print(c)
    _colorize_object(ob,c,False)



bpy.ops.wm.save_as_mainfile(filepath=f"{args.save_tmp_blend}")
raise()



frames = []
obj_camera = cam
# raise()
for i_pos, look_data in enumerate(look_at_trans):
    print(look_data)
    obj_camera.location = (
        (look_data['eye'][0])+look_data['at'][0],
        (look_data['eye'][1])+look_data['at'][1],
        (look_data['eye'][2])+look_data['at'][2]
        )
    look_at(obj_camera,look_data['eye'])

    bpy.context.view_layer.update()

    # export_to_ndds_file(
    #     f"{path}/{str(i_pos).zfill(5)}.json",
    #     obj_names = ,
    #     width = args.resolution,
    #     height = args.resolution,
    #     camera_name = 'camera',
    #     cuboids = cuboids,
    #     camera_struct = camera_struct,
    #     segmentation_mask = np.array(segmentation_array).reshape(cfg.height,cfg.width,4)[:,:,0],
    #     scene_aabb = scene_aabb,
    # )
    

    rt = get_3x4_RT_matrix_from_blender(obj_camera)
    pos, rt, scale = obj_camera.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for i in range(3):
        a = []
        for j in range(3):
            a.append(rt[i][j])
        a.append(pos[i])
        matrix.append(a)
    matrix.append([0,0,0,1])
    # print(matrix)

    # print(obj_camera.matrix_world.decompose())
    # raise()
    # matrix[0][-1]=obj_camera.location[0]
    # matrix[1][-1]=obj_camera.location[1]
    # matrix[2][-1]=obj_camera.location[2]
    # print(matrix)

    to_add = {\
        "file_path":f'{str(i_pos).zfill(3)}.png',
        "transform_matrix":matrix
    }
    frames.append(to_add)

    bpy.context.scene.render.filepath = f'{path}/{str(i_pos).zfill(3)}.png'
    bpy.ops.render.render(write_still = True)
    # raise()
    # time.sleep(10)
    # break

    to_export['frames'] = frames

with open(f'{path}/transforms.json', 'w') as f:
    json.dump(to_export, f,indent=4)