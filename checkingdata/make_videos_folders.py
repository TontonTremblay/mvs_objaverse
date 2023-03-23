import glob 
import subprocess

for folder in glob.glob("/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/handal_hammer_syn/*/"):
	name = folder.split('/')[-2]
	subprocess.call(
		[
			"python",'../github_nvisii_mvs/visualization/make_video_folder2.py',
			'--path', folder,
			'--outf',f'{name}.mp4'
		])