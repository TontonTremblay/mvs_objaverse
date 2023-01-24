import multiprocessing
import objaverse

processes = multiprocessing.cpu_count()

import random

random.seed(12)

uids = objaverse.load_uids()
random_object_uids = random.sample(uids, 10)

# the assets are stored in /home/USERNAME/.objaverse
objects = objaverse.load_objects(
    uids=random_object_uids,
    download_processes=processes
)
