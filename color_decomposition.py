from os import walk
import os
import re
import cv2
import numpy as np
from PIL import Image
from dominant_color_detection import detect_colors
from posterization import preprocess, hex2rgb, save_palette, get_matrix, save_cluster, save_json, get_cluster_data, data_score_mult

PATH_TO_DIR = 'images'

fnames = []
for (dirpath, dirnames, filenames) in walk(PATH_TO_DIR):
	fnames.extend(filenames)
	break

images = [filename for filename in fnames if 'axis' not in filename and 'Legend' not in filename and 'png' in filename]
legends = [filename for filename in fnames if 'Legend' in filename and 'json' in filename]

for image_name in images:
	print(image_name)
	legend_names = [os.path.join(PATH_TO_DIR,legend) for legend in legends if re.match(image_name[:-4]+'_Legend', legend)]
	image_arr = preprocess(os.path.join(PATH_TO_DIR,image_name), legend_names)
	if (image_arr<255).any():
		image = Image.fromarray(image_arr)
		image.save(os.path.join(PATH_TO_DIR,image_name)[:-4]+'_colorcut.png')
		colors, _ = detect_colors(os.path.join(PATH_TO_DIR,image_name)[:-4]+'_colorcut.png')
		colorsrgb = [hex2rgb(colors[i]) for i in range(len(colors))]
		save_palette([colorsrgb],os.path.join(PATH_TO_DIR,image_name))
		matrix = get_matrix(image_arr,colorsrgb)
		for i in range(1,5):
			cluster = get_cluster_data(matrix,i)
			if len(cluster)>100:
				score_m = data_score_mult(cluster)
				if score_m > 0.66:
					save_cluster(cluster,i,colors[i],os.path.join(PATH_TO_DIR,image_name))
					save_json(cluster,colors[i],i,os.path.join(PATH_TO_DIR,image_name))
