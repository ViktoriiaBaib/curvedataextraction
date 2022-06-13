from os import walk
import os
import re
import cv2
import numpy as np
from PIL import Image
from posterization import detect_colors
from posterization import preprocess, rgb2hex, save_palette, get_matrix, save_cluster, save_json, get_cluster_data, data_score_mult

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
		colorsrgb = detect_colors(image_arr)
		#print('colors: ',colorsrgb, type(colorsrgb))
		save_palette([colorsrgb],os.path.join(PATH_TO_DIR,image_name))
		matrix = get_matrix(image_arr,colorsrgb)
		for i in range(len(colorsrgb)):
			cluster = get_cluster_data(matrix,i)
			if len(cluster)>300:
				score_m = data_score_mult(cluster)
				#print('Cluster '+str(i)+' score: ', score_m)
				if score_m > 0.66:
					save_cluster(cluster,i,rgb2hex(colorsrgb[i]),os.path.join(PATH_TO_DIR,image_name))
					save_json(cluster,rgb2hex(colorsrgb[i]),i,os.path.join(PATH_TO_DIR,image_name))
