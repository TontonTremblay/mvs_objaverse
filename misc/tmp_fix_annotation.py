import json 
import glob 
import os 

imgs = []

def add_json_files(path):
    global imgs
    for imgpath in sorted(glob.glob(path+"/*.png")):
        # if 'rgb' in imgpath:
        imgs.append([imgpath])
    for imgpath in sorted(glob.glob(path+"/*.jpg")):
        # imgs.append([imgpath])
        # if 'rgb' in imgpath:
        imgs.append([imgpath])
    for imgpath in sorted(glob.glob(path+"/*.exr")):
        # print(imgpath)
        # if "depth" in imgpath or "seg" in imgpath or 'nerf' in imgpath:
        if "depth" in imgpath or "seg" in imgpath:
            continue
        # imgs.append([imgpath])
        # if 'rgb' in imgpath:
        imgs.append([imgpath,imgpath.replace('exr','seg.exr')])
        

def explore(path):
    global imgs
    if not os.path.isdir(path):
        return
    folders = [os.path.join(path, o) for o in os.listdir(path) 
                    if os.path.isdir(os.path.join(path,o))]
    if len(folders)>0:
        for path_entry in folders:                
            explore(path_entry)
        # add_json_files(path)
    # else:
    add_json_files(path)


explore("/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/handal_hammer_syn/")


for imgpath in imgs:
    json_path = imgpath[0].replace("png",'json').replace('exr','json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    for obj in data['objects']:
        old_proj = obj['projected_cuboid']
        new_proj = []
        new_proj.append(old_proj[0])
        new_proj.append(old_proj[4])
        new_proj.append(old_proj[5])
        new_proj.append(old_proj[1])
        new_proj.append(old_proj[3])
        new_proj.append(old_proj[7])
        new_proj.append(old_proj[6])
        new_proj.append(old_proj[2])
        new_proj.append(old_proj[8])
        obj['projected_cuboid'] = new_proj
        obj['old_projected_cuboid'] = old_proj
        
        old_proj = obj['local_cuboid']
        new_proj = []
        new_proj.append(old_proj[0])
        new_proj.append(old_proj[4])
        new_proj.append(old_proj[5])
        new_proj.append(old_proj[1])
        new_proj.append(old_proj[3])
        new_proj.append(old_proj[7])
        new_proj.append(old_proj[6])
        new_proj.append(old_proj[2])
        new_proj.append(old_proj[8])
        obj['local_cuboid'] = new_proj
        obj['old_local_cuboid'] = old_proj        
    with open(json_path, 'w+') as fp:
        json.dump(data, fp, indent=4, sort_keys=True)
    # raise()