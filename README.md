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

The output from the script is compatible with instant-ngp. And here is a sample for the mugs: 
![renders](https://i.imgur.com/CcdGXJL.jpg)


# Animations 
Maybe you want to animate some renders e.g., 360 view. 

Check `render_all_models.py`

### Falling scene 

Check `rendering/render_faling_hammers.py`. You are going to need to download the following assets and make sure the paths are set correctly. 
- [hammer_assets](https://drive.google.com/drive/folders/1eZnGriYr2e8vmUfowo00Uc3VF0OTVXtk?usp=share_link)
- [HDRI env. map](https://drive.google.com/file/d/1lp36MgTlS4OFaH0vdsTFhyGFJpQDY2YX/view?usp=share_link)
- [cco textures](https://drive.google.com/file/d/1GWpRqSn_GKn0fwfEHFpQctfEo51KiqbY/view?usp=share_link)
- [google scanned assets](https://drive.google.com/drive/folders/1i-4NzkhNY2gfMXb--yAePPFzg1RMoa-I?usp=share_link)

Then you run the script `python rendering/render_faling_hammers.py` with the correct path to the assets and to your blender install. 

If you want to generate a dataset, check `rendering/render_dataset_falling_hammer.py`.

# TODO 
- Share some more complete renders (google drive links) 
