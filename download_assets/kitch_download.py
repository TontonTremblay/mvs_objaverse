import multiprocessing
import objaverse
import random
import subprocess
import glob 

processes = multiprocessing.cpu_count()

lvis_annotations = objaverse.load_lvis_annotations()

data_to_load = ["alcohol","barrel","beanbag","beer_bottle","beer_can","blender","bottle","bottle_opener","bowl","box","can_opener","cappuccino","casserole","cayenne_(spice)","chopping_board","chopstick","cocoa_(beverage)","coffee_maker","colander","condiment","cooker","cork_(bottle_plug)","cream_pitcher","cup","cupboard","cupcake","detergent","dish","dishrag","dishtowel","dustpan","eggbeater","first-aid_kit","flowerpot","food_processor","fork","fruit_juice","frying_pan","gravy_boat","hot_sauce","ice_maker","icecream","martini","measuring_cup","milk","milk_can","mug","nutcracker","olive_oil","pan_(for_cooking)","pan_(metal_container)","pitcher_(vessel_for_liquid)","salad_plate","saucepan","shears","soup_bowl","soya_milk","spice_rack","steak_knife","sugar_bowl","tablecloth","thermos_bottle","thermometer","toaster","toaster_oven","tongs","toolbox","urn","vase","water_bottle","water_cooler","water_jug","watering_can","whipped_cream","wine_bottle","wine_bucket","wineglass"]

where_to_store = "/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/objaverse/"

for key in data_to_load:
	print(key)
	objects = objaverse.load_objects(
	    uids=lvis_annotations[key],
	    download_processes=processes
	)
	subprocess.call(['mkdir',f'{where_to_store}/{key}/'])
	for folder in glob.glob('/home/jtremblay/.objaverse/hf-objaverse-v1/glbs/*/'):
		subprocess.call(['mv',folder,f'{where_to_store}/{key}/'])
	# break
