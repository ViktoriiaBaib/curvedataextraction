import easyocr
import cv2
from easyocr import Reader
import json
import numpy as np
import pandas as pd
import re
from os import walk
import os
import argparse
from final_record_func import get_scaling, save_json

parser = argparse.ArgumentParser()
parser.add_argument('-x','--xunits', default='um', type=str, help='Units of measurement for X axis. Options are: um, cm-1, eV, nm, eV/h_bar')
#parser.add_argument('-y','--yunits', default='1', type=str, help='Units of measurement for Y axis. Options are: 1, au')
args = parser.parse_args()
xaxunits = args.xunits

reader = easyocr.Reader(['en'], gpu=True)
allowlist = '0123456789.-'
PATH_TO_DIR = os.getcwd()
print(PATH_TO_DIR)

fnames = []
for (dirpath, dirnames, filenames) in walk(os.path.join(PATH_TO_DIR,'images')):
    fnames.extend(filenames)
    break
#print(anames)
imgnames = [filename for filename in fnames if 'png' in filename and 'axis' not in filename and 'Legend' not in filename and 'colorcut' not in filename]
#print(imgnames)
axisnames = [filename for filename in fnames if 'axis' in filename and 'json' in filename]
#print(axisnames)
clusternames = [filename for filename in fnames if 'cluster' in filename and 'json' in filename]
#print(clusternames)


for i,clustername in enumerate(clusternames):
	FILE = clustername[:-24]
	print(FILE)
	f = open(os.path.join('images',clustername))  
	cluster = json.load(f)
	f.close()
	AXIS = [xname for xname in axisnames if re.match(FILE+'_X_axis', xname)][0]
	print(AXIS)
	f = open(os.path.join('images',AXIS))
	print(os.path.join('images',AXIS))
	x_axis_json = json.load(f)
	f.close()
	x_axis_img = cv2.imread(os.path.join('images',AXIS[:-4]+'png'))
	print(os.path.join('images',AXIS[:-4]+'png'))
	[confidence, a1, x_box] = get_scaling(x_axis_json, x_axis_img, reader, allowlist)
	if confidence == 'unconfident':
		print(i, clustername, FILE, confidence)
		continue
	else:
		save_json(FILE, clustername, cluster, x_box, a1, xaxunits)
		print(i, clustername, FILE, confidence)

