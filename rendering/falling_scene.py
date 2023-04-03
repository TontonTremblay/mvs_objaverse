import argparse, sys, os, math, re
import bpy
from mathutils import Vector, Matrix
import mathutils
import numpy as np
import json 
import random 
import glob 
# from PIL import Image
# import png 
import threading

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
    '--save_tmp_blend', type=str, default='/home/jtremblay/code/mvs_objaverse/tmp.blend',
    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

parser.add_argument(
    '--nb_objects_cat', type=int, default=5,
    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

parser.add_argument(
    '--distractors', type=int, default=5,
    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

parser.add_argument(
    '--save_segmentation', action="store_true",
    help='render segmentation as _seg.exr')

parser.add_argument(
    '--save_depth', action="store_true",
    help='render depth as _depth.exr')



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
    scaling_value = random.uniform(1, 3)
    bpy.context.object.scale = (scaling_value,scaling_value,1)
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.context.object.rigid_body.collision_margin = 0

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

    scaling_value *= 0.5

    ob = bpy.ops.mesh.primitive_plane_add(size=2, 
        enter_editmode=False, 
        align='WORLD', 
        location=(0, -scaling_value, 10), 
        scale=(1, 1, 1)
    )

    bpy.context.object.scale = (scaling_value,scaling_value,1)
    bpy.context.object.rotation_mode = 'XYZ'
    bpy.context.object.rotation_euler = (np.pi/2,0,0)

    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.context.object.rigid_body.collision_margin = 0
    bpy.context.object.hide_render = True
    bpy.context.object.hide_viewport = True


    ob = bpy.ops.mesh.primitive_plane_add(size=2, 
        enter_editmode=False, 
        align='WORLD', 
        location=(0, scaling_value, 10), 
        scale=(1, 1, 1)
    )

    bpy.context.object.scale = (scaling_value,scaling_value,1)
    bpy.context.object.rotation_mode = 'XYZ'
    bpy.context.object.rotation_euler = (np.pi/2,0,0)

    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.context.object.rigid_body.collision_margin = 0
    bpy.context.object.hide_render = True
    bpy.context.object.hide_viewport = True


    ob = bpy.ops.mesh.primitive_plane_add(size=2, 
        enter_editmode=False, 
        align='WORLD', 
        location=(-scaling_value, 0, 10), 
        scale=(1, 1, 1)
    )

    bpy.context.object.scale = (scaling_value,scaling_value,1)
    bpy.context.object.rotation_mode = 'XYZ'
    bpy.context.object.rotation_euler = (0,np.pi/2,0)

    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.context.object.rigid_body.collision_margin = 0
    bpy.context.object.hide_render = True
    bpy.context.object.hide_viewport = True

    ob = bpy.ops.mesh.primitive_plane_add(size=2, 
        enter_editmode=False, 
        align='WORLD', 
        location=(scaling_value, 0, 10), 
        scale=(1, 1, 1)
    )

    bpy.context.object.scale = (scaling_value,scaling_value,1)
    bpy.context.object.rotation_mode = 'XYZ'
    bpy.context.object.rotation_euler = (0,np.pi/2,0)

    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.collision_shape = 'MESH'
    bpy.context.object.rigid_body.collision_margin = 0
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
def LookAt(obj, target, roll=0):
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
    height = 500, 
    width = 500,
    camera_ob = 'my_camera',
    camera_struct = None,
    data = None,
    segmentation_mask = None,
    scene_aabb = []
    ):
    # To do export things in the camera frame, e.g., pose and quaternion


    # import simplejson as json
    import json
    # cam_world_quaternion = visii.quat_cast(cam_matrix)

    cam_intrinsics = get_calibration_matrix_K_from_blender(camera_ob.data)

    if camera_struct is None:
        camera_struct = {
            'at': [0,0,0,],
            'eye': [0,0,0,],
            'up': [0,0,0,]
        }


    rt = get_3x4_RT_matrix_from_blender(obj_camera)
    pos, rt, scale = obj_camera.matrix_world.decompose()
    rt = rt.to_matrix()
    cam2wold = []
    for i in range(3):
        a = []
        for j in range(3):
            a.append(rt[i][j])
        a.append(pos[i])
        cam2wold.append(a)
    cam2wold.append([0,0,0,1])

    cam_world_location = obj_camera.location 

    cam_world_quaternion = obj_camera.rotation_euler.to_quaternion()

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
                    'cam2world':cam2wold,
                    'location_world':
                    [
                        cam_world_location[0],
                        cam_world_location[1],
                        cam_world_location[2],
                    ],
                    'intrinsics':{
                        'fx':cam_intrinsics[0][0],
                        'fy':cam_intrinsics[1][1],
                        'cx':cam_intrinsics[2][0],
                        'cy':cam_intrinsics[2][1]
                    },
                    # 'scene_min_3d_box':scene_aabb[0],
                    # 'scene_max_3d_box':scene_aabb[1],
                    # 'scene_center_3d_box':scene_aabb[2],
                }, 
                "objects" : []
            }

    # load the segmentation & find unique pixels
    segmentation_mask = bpy.data.images.load(f'{path}/{str(i_pos).zfill(3)}_seg.exr')
    segmentation_mask = np.asarray(segmentation_mask.pixels)
    segmentation_mask = segmentation_mask.reshape((args.resolution,args.resolution,4))[:,:,:3]
    unique_pixels = np.vstack({tuple(r) for r in segmentation_mask.reshape(-1,3)})
    unique_pixels = (unique_pixels*255).astype(int)


    # Segmentation id to export
    import bpy_extras
    for obj_name in data.keys(): 
        projected_keypoints = []
        
        obj = bpy.context.scene.objects[obj_name]

        for keypoint in bpy.context.scene.objects[obj_name].children:
            if not keypoint.type == "EMPTY":
                continue

            co_2d = bpy_extras.object_utils.world_to_camera_view(
                bpy.context.scene, 
                camera_ob, 
                keypoint.matrix_world.translation
            )

            # If you want pixel coords
            render_scale = scene.render.resolution_percentage / 100
            render_size = (int(scene.render.resolution_x * render_scale),
                           int(scene.render.resolution_y * render_scale),
                        )
            projected_keypoints.append([co_2d.x * render_size[0],height - co_2d.y * render_size[1]])
 
        cuboid = data[obj_name]['cuboid3d']

        #check if the object is visible
        visibility = -1
        bounding_box = [-1,-1,-1,-1]


        # Using the bouding box keypoints wont give tight 2d bounding box
        # use the segmentation mask instead. 
        a = np.array(projected_keypoints)
        minx = min(a[:,0])
        miny = min(a[:,1])
        maxx = max(a[:,0])
        maxy = max(a[:,1])

        # Not working
        # if (minx>0 and minx<width and miny>0 and miny<height ) or\
        #    (maxx>0 and maxx<width and maxy>0 and maxy<height ) or\
        #    (minx>0 and minx<width and maxy>0 and maxy<height ) or\
        #    (maxx>0 and maxx<width and miny>0 and miny<height ):
        #    visibility = 1
        # else:
        #     visibility = 0 

        color_int = (np.array(data[obj_name]['color_seg'])*255).astype(int)
        if color_int in unique_pixels:
            visibility = 1
        else:
            visibility = 0

        pos, rt, scale = obj.matrix_world.decompose()
        rt = rt.to_matrix()
        trans_matrix_export = []
        for i in range(3):
            a = []
            for j in range(3):
                a.append(rt[i][j])
            a.append(pos[i])
            trans_matrix_export.append(a)
        trans_matrix_export.append([0,0,0,1])
        
        rt = obj_camera.convert_space(matrix=obj.matrix_world, to_space='LOCAL')
        trans_matrix_cam_export = []
        for i in range(3):
            a = []
            for j in range(3):
                a.append(rt[i][j])
            a.append(pos[i])
            trans_matrix_cam_export.append(a)
        trans_matrix_cam_export.append([0,0,0,1])
        

        # Final export
        dict_out['objects'].append({
            # 'class':obj_name.split('_')[1],
            'name':obj.name,
            'provenance':'blender_john',
            # TODO check the location
            'location_world': [
                obj.location[0],
                obj.location[1],
                obj.location[2]
            ],
            'local_to_world_matrix':trans_matrix_export,
            'trans_in_camera':trans_matrix_cam_export,
            'projected_cuboid':projected_keypoints,
            'local_cuboid': cuboid,
            'visibility':visibility,
            'bounding_box_minx_maxx_miny_maxy':[minx,maxx,miny,maxy],
        })
        dict_out['objects'][-1]['segmentation_id']=data[obj_name]["color_seg"]


    with open(filename, 'w+') as fp:
        json.dump(dict_out, fp, indent=4, sort_keys=True)
    # return bounding_box

def EmissionColorObj(obj, color):
    new_mat = bpy.data.materials.new(name="seg")
    new_mat.use_nodes = True
    new_mat.cycles.sample_as_light = False
    emission_node = new_mat.node_tree.nodes.new(type='ShaderNodeEmission')
    output = new_mat.node_tree.nodes.get("Material Output")

    emission_node.inputs['Color'].default_value[:3] = color
    new_mat.node_tree.links.new(emission_node.outputs['Emission'], output.inputs['Surface'])

    if len(obj.material_slots) > 0:
        for imat,mat in enumerate(obj.material_slots):
                obj.data.materials[imat] = new_mat
    else:
        obj.data.materials.append(new_mat)

def AddCuboid(obj_parent): 
    # find the children

    def get_all_child(ob,to_return = []):
        if len(ob.children) == 0: 
            return [ob]
        to_add = []
        for child in ob.children:
            to_add += [ob]
            to_add += get_all_child(child)
        return to_add

    mesh_objs = []
    for obj in get_all_child(obj_parent):
        print(obj.name,obj.type)
        if not obj.type == 'MESH':
            continue
        mesh_objs.append(obj)
    
    corners = []
    
    for ob in mesh_objs:
        print(ob.name)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bbox_corners = [ob.matrix_world @ Vector(corner)  for corner in ob.bound_box]
        bbox_corners = [Vector(corner)  for corner in ob.bound_box]
        
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
    max_obj = maxb
    min_obj = mina
    centroid_obj = center_point

    cuboid = [
        (max_obj[0], min_obj[1], max_obj[2]),
        (min_obj[0], min_obj[1], max_obj[2]),
        (min_obj[0], max_obj[1], max_obj[2]),
        (max_obj[0], max_obj[1], max_obj[2]),
        (max_obj[0], min_obj[1], min_obj[2]),
        (min_obj[0], min_obj[1], min_obj[2]),
        (min_obj[0], max_obj[1], min_obj[2]),
        (max_obj[0], max_obj[1], min_obj[2]),
        (centroid_obj[0], centroid_obj[1], centroid_obj[2]),
    ]    

    for ip, p in enumerate(cuboid):
        bpy.ops.object.empty_add(location=p)
        ob = bpy.context.object
        ob.name = f'{ip}_{obj_parent.name}'
        ob.parent = obj_parent

    return cuboid

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


NB_OBJECTS_LOADED = random.randint(max(1,args.nb_objects_cat-2),max(2,args.nb_objects_cat+2))

NB_OBJECTS_LOADED_OTHERS = 0 
if args.distractors >0:
    NB_OBJECTS_LOADED_OTHERS = random.randint(max(1,args.distractors-2),max(2,args.distractors+2))

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

DATA_2_EXPORT = {}



if args.input_model == "glb":
    assets_content = glob.glob(f"{args.folder_assets}/*.{args.input_model}")

    for i in range(NB_OBJECTS_LOADED): 
        to_load = assets_content[random.randint(0,len(assets_content)-1)]
        print(to_load)
        name = to_load.split("/")[-1].split(".")[0] + "_" + str(i).zfill(2)
        imported_object = bpy.ops.import_scene.gltf(filepath=to_load)

        for ob in bpy.context.selected_objects:
            if ob.type == 'MESH':
                break
        bpy.ops.rigidbody.object_add({'object': ob})
        ob.rigid_body.collision_shape = 'CONVEX_HULL'
        ob.rigid_body.use_margin = True
        ob.rigid_body.collision_margin = 0

        ob.name = name
        cuboid3d = AddCuboid(ob)
        DATA_2_EXPORT[ob.name] = {}
        DATA_2_EXPORT[ob.name]['cuboid3d']=cuboid3d
        # add the cuboid 

# bpy.ops.wm.save_as_mainfile(filepath=f"{args.save_tmp_blend}")
# raise()

# load some distractors 

obj_to_export = []
pos = 0.5
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
        ob.location = (random.uniform(-pos,pos),random.uniform(-pos,pos),random.uniform(pos+1,pos+6))
        # ob.select_set(True)
        # bpy.data.scenes['Scene'].rigidbody_world.collection.objects.link(ob)
        # bpy.ops.rigidbody.object_add()

        if 'model' in ob.name:
            s = random.uniform(0.2,1)
            s = 1
            ob.scale = (s,s,s)
            bpy.ops.rigidbody.object_add({'object': ob})
            ob.rigid_body.collision_shape = 'CONVEX_HULL'
            ob.rigid_body.use_margin = True
            ob.rigid_body.collision_margin = 0

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
# scene.frame_set(0)
bpy.context.view_layer.update()

# for jj in range(250):
bpy.context.scene.rigidbody_world.substeps_per_frame = 10
bpy.context.scene.rigidbody_world.solver_iterations = 10
bpy.context.scene.rigidbody_world.enabled = True

# Run simulation
point_cache = bpy.context.scene.rigidbody_world.point_cache
point_cache.frame_start = 1

frame_set = 60
for i in range(frame_set):
    point_cache.frame_end = i
    bpy.ops.ptcache.bake({"point_cache": point_cache}, bake=True)
    scene.frame_set(i)
    bpy.context.view_layer.update()

#
bpy.ops.wm.save_as_mainfile(filepath=f"{args.save_tmp_blend.replace('.blend','-1.blend')}")



# generate camera poses 
factor = random.uniform(1,5)
cfg = {}
cfg['camera_nb_anchor'] = random.randint(30, 100)
cfg['camera_elevation_range'] = [0,360]
cfg['camera_theta_range'] = [20,80]
cfg['to_add_position'] = [random.randint(-5, 5),random.randint(-5, 5),factor+0.5]
cfg['camera_fixed_distance_factor'] = factor
cfg['camera_nb_frames'] = args.views
cfg = dotdict(cfg)

# anchor_points = random_sample_sphere(
#                         nb_frames = cfg.camera_nb_anchor,
#                         elevation_range = cfg.camera_elevation_range,
#                         tetha_range = cfg.camera_theta_range
#                     )
# anchor_points[-1] = anchor_points[0]

# t_points = np.arange(0, 1, 1/cfg.camera_nb_frames) #................................. Creates an iterable list from 0 to 1.
# anchor_points = np.array(anchor_points)
# positions_to_render = Bezier.Curve(t_points, anchor_points) #......................... Returns an array of coordinates.

# at_all = []
# at_name = list(DATA_2_EXPORT.keys())[np.random.randint(0,len(list(DATA_2_EXPORT.keys())))]
# pos = bpy.data.objects[at_name].location
# at = [pos[0],pos[1],pos[2]]
# for i_at in range(len(anchor_points)):
#     if random.random() < 0.5:
#         at_name = list(DATA_2_EXPORT.keys())[np.random.randint(0,len(list(DATA_2_EXPORT.keys())))]
#         pos = bpy.data.objects[at_name].location
#         at = [pos[0],pos[1],pos[2]]
#     # at = [0,0,0]
#     at_all.append(at)
# at_all[-1] = at_all[0]    
# at_all = np.array(at_all)
# # print(at_all)
# at_all = Bezier.Curve(t_points, at_all) #......................... Returns an array of coordinates.


# # generate the camera poses 
# look_at_trans = []
# to_add = [0,0,0]
# if "to_add_position" in cfg:
#     to_add = cfg.to_add_position
# for i_pos, pos in enumerate(positions_to_render):
#     look_at_trans.append({
#         'at': at_all[i_pos],
#         'up': [0,0,1],
#         'eye': [pos[0]*float(cfg.camera_fixed_distance_factor)+to_add[0],
#                 pos[1]*float(cfg.camera_fixed_distance_factor)+to_add[1],
#                 pos[2]*float(cfg.camera_fixed_distance_factor)+to_add[2]]              
#         }
#     )

# positions_to_render = random_sample_sphere(
#                             nb_frames = cfg.camera_nb_frames,
#                             elevation_range = cfg.camera_elevation_range,
#                             tetha_range = cfg.camera_theta_range
#                         )
positions_to_render = []
for ipos in range(cfg.camera_nb_frames):
    positions_to_render.append(
        [
            random.uniform(-1,1),
            random.uniform(-1,1),
            random.uniform(0,1),
        ]
    )    


look_at_trans = []
for pos in positions_to_render:

    at_name = list(DATA_2_EXPORT.keys())[np.random.randint(0,len(list(DATA_2_EXPORT.keys())))]
    pos_obj= bpy.data.objects[at_name].location
    look_at = [pos_obj[0],pos_obj[1],pos_obj[2]]
    # raise()
    # print(look_at)
    look_at_trans.append({
        'at': look_at,
        'up': [0,0,1],
        # 'eye': [pos[0]*float(cfg.camera_fixed_distance_factor),
        #         pos[1]*float(cfg.camera_fixed_distance_factor),
        #         pos[2]*float(cfg.camera_fixed_distance_factor)]              
        'eye': [pos[0]*random.uniform(0.1,1),
                pos[1]*random.uniform(0.1,1),
                pos[2]*random.uniform(0.1,1)]              

        })
    print(look_at)
# raise()
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
# result = bpy.ops.scene.new(type='LINK_COPY')
bpy.context.scene.name = "segmentation"


# lets update all the materials of the objects to emmissive
to_change = []
for ob in bpy.context.scene.objects:
    if ob.type == 'MESH':
        to_change.append(ob)

import colorsys

for ob in to_change:
    while True:
        c = colorsys.hsv_to_rgb(
            random.uniform(0,255)/255, 
            random.uniform(200,255)/255, 
            random.uniform(200,255)/255
            )
        found = False
        for obj in DATA_2_EXPORT:
            if 'color_seg' in DATA_2_EXPORT[obj] and c == DATA_2_EXPORT[obj]['color_seg']:
                found = True
        if found is True:
            continue
        if ob.name.split(".")[0] in DATA_2_EXPORT:
            DATA_2_EXPORT[ob.name.split(".")[0]]['color_seg'] = c
        break
    EmissionColorObj(ob,c)

nodes = bpy.context.scene.world.node_tree.nodes
links = bpy.context.scene.world.node_tree.links

c = colorsys.hsv_to_rgb(
    random.uniform(0,255)/255, 
    random.uniform(200,255)/255, 
    random.uniform(200,255)/255
    )
c = [c[0],c[1],c[2],1]
if len(nodes.get("Background").inputs['Color'].links) > 0:
    links.remove(nodes.get("Background").inputs['Color'].links[0])

nodes.get("Background").inputs['Strength'].default_value = 1
nodes.get("Background").inputs['Color'].default_value = c 





bpy.context.scene.cycles.samples =1
bpy.context.view_layer.cycles.use_denoising = False
bpy.context.scene.render.use_motion_blur = False
bpy.context.scene.render.image_settings.file_format="OPEN_EXR"
bpy.context.scene.render.image_settings.compression=0
bpy.context.scene.render.image_settings.color_mode="RGBA"
bpy.context.scene.render.image_settings.color_depth="32"
bpy.context.scene.render.image_settings.exr_codec="NONE"
bpy.context.scene.render.image_settings.use_zbuffer=True
bpy.context.view_layer.use_pass_z=True


bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

for n in tree.nodes:
    tree.nodes.remove(n)

# Create input render layer node.
render_layers = tree.nodes.new('CompositorNodeRLayers')

depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
depth_file_output.format.file_format = "OPEN_EXR"
depth_file_output.base_path = f'{path}'

node_viewer = tree.nodes.new('CompositorNodeViewer') 
node_viewer.use_alpha = False  
links.new(render_layers.outputs['Image'], node_viewer.inputs[0])

frames = []
obj_camera = cam
# raise()


bpy.context.window.scene = bpy.data.scenes['Scene']

bpy.ops.wm.save_as_mainfile(filepath=f"{args.save_tmp_blend}")


for i_pos, look_data in enumerate(look_at_trans):
    # print(look_data)

    bpy.context.window.scene = bpy.data.scenes['segmentation']
    obj_camera = bpy.context.scene.objects['Camera.001']
    obj_camera.location = (
        (look_data['eye'][0]),
        (look_data['eye'][1]),
        (look_data['eye'][2])
        )
    bpy.context.view_layer.update()
    LookAt(obj_camera,look_data['at'])

    bpy.context.view_layer.update()
    depth_file_output.file_slots[0].path = f'{str(i_pos).zfill(3)}_depth'
    bpy.context.scene.render.filepath = f'{path}/{str(i_pos).zfill(3)}_seg.exr'
    
    bpy.ops.render.render(write_still = True)    
    os.rename(f'{path}/{str(i_pos).zfill(3)}_depth{str(frame_set-1).zfill(4)}.exr', f'{path}/{str(i_pos).zfill(3)}_depth.exr') 

    # pixels = bpy.data.images['Viewer Node'].pixels
    
    # segmentation_mask = np.asarray(pixels)
    # segmentation_mask = segmentation_mask.reshape((args.resolution,args.resolution,4))
    # print(pixels)





    # render.engine = args.engine
    bpy.context.window.scene = bpy.data.scenes['Scene']
    obj_camera = bpy.context.scene.objects['Camera']
    obj_camera.location = (
        (look_data['eye'][0]),
        (look_data['eye'][1]),
        (look_data['eye'][2])
        )
    bpy.context.view_layer.update()
    
    LookAt(obj_camera,look_data['at'])

    bpy.context.view_layer.update()

    # change the scene 


    export_to_ndds_file(
        f"{path}/{str(i_pos).zfill(3)}.json",
        width = args.resolution,
        height = args.resolution,
        camera_ob = obj_camera,
        data = DATA_2_EXPORT,
        camera_struct = look_data,
        segmentation_mask = f'{path}/{str(i_pos).zfill(3)}_seg.png',
        scene_aabb = [],
    )
    
    # if no depth or no segmentation to be saved.
    if not args.save_segmentation:
        os.remove(f'{path}/{str(i_pos).zfill(3)}_seg.exr')
        # threading.Thread(os.remove, f'{path}/{str(i_pos).zfill(3)}_seg.exr').start()
    if not args.save_depth:
        os.remove(f'{path}/{str(i_pos).zfill(3)}_depth.exr')
        # threading.Thread(os.remove, f'{path}/{str(i_pos).zfill(3)}_depth.exr').start()


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

    to_export['frames'] = frames

with open(f'{path}/transforms.json', 'w') as f:
    json.dump(to_export, f,indent=4)