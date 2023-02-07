# How to download the assets 
[Objaverse](https://objaverse.allenai.org/#explore) has a nice python API to download assets from [sketchfab](https://sketchfab.com/). 
In `download_assets` there are a couple examples to download the assets. The assets are stored in `/home/USERNAME/.objaverse/hf-objaverse-v1/glbs/`. 

# Installation
Download [Blender](https://www.blender.org/), this code base uses Blender **3.2.0**. 
Also make sure to `pip install -r requirements.txt`

# Rendering assets for MVS 
The blender code is inspired from [wisp](https://drive.google.com/drive/folders/1Via1TOsnG-3mUkkGteEoRJdEYJEx3wgf) and [get3d](https://github.com/nv-tlabs/GET3D/tree/master/render_shapenet_data).


Here is a complete workflow for rendering, please note that everything has to be in absolute path. 
```
python download_assets/random_samples.py # saves 3d models in USERNAME/.objaverse/hf-objaverse-v1/glbs/
mkdir output
python rendering/render_all_models.py --save_folder /home/jtremblay/code/mvs_objaverse/output/ --folder_assets /home/jtremblay/.objaverse/hf-objaverse-v1/glbs/ --blender_root /home/jtremblay/Desktop/blender-3.2.0-alpha+master.e2e4c1daaa47-linux.x86_64-release/blender
```

The output from the script is compatible instant-ngp. And here is a sample for the mugs: 
![renders](https://i.imgur.com/CcdGXJL.jpg)

# TODO 
- Share some more complete renders (google drive links) 
