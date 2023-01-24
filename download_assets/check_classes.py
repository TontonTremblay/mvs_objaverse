import multiprocessing
import objaverse

processes = multiprocessing.cpu_count()

import random

lvis_annotations = objaverse.load_lvis_annotations()
print(len(lvis_annotations['apple']))

