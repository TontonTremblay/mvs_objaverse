import multiprocessing
import objaverse

processes = multiprocessing.cpu_count()

import random

random.seed(42)

uids = objaverse.load_uids()
random_object_uids = random.sample(uids, 10)

# objects = objaverse.load_objects(
#     uids=random_object_uids,
#     download_processes=processes
# )

lvis_annotations = objaverse.load_lvis_annotations()
print(f"nb_mugs: {len(lvis_annotations)}")

objects = objaverse.load_objects(
    uids=lvis_annotations['mug'],
    download_processes=processes
)