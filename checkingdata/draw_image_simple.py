import cv2 
import json 

path = 'output/tmp/000.png'

im = cv2.imread(path)

f = open(path.replace('png','json'))
  
# returns JSON object as 
# a dictionary
data = json.load(f)

for obj in data['objects']:
	for p in obj['projected_cuboid']:
		im = cv2.circle(im, (int(p[0]),int(p[1])), 3, (0,255,0), 0)
	# break
cv2.imwrite('tmp2.png',im)