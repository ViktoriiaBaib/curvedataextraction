import easyocr
import cv2
from easyocr import Reader
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import math
import re
from os import walk
import os
'''
def rgb_to_hex(color):
    [r,g,b] = color
    rgb = (int(np.round(255*r)), int(np.round(255*g)), int(np.round(255*b)))
    return '#%02x%02x%02x' % rgb
'''
def get_scaling(x_axis_json, x_axis_img, reader, allowlist):
    x_box = x_axis_json['detected_box'][0]
    x_width = x_box['xmax']-x_box['xmin']
    x_height = x_box['ymax']-x_box['ymin']
    x_axis_img = cv2.copyMakeBorder(x_axis_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT,value=(255,255,255))
    x_results = reader.readtext(x_axis_img, min_size = 10, mag_ratio=2, allowlist=allowlist)
    a = clean_array(x_results)
    [a1, confidence] = well_approximated(a)
    return [confidence, a1, x_box]

def clean_array(x_results):
    a = np.array([])
    record = []
    height_box = 10
    for (bbox, text, prob) in x_results:
        if prob < 0.5:
            continue
        if text =='' or text == '.' or text == '-':
            continue
        else:
            try:
                float(text)
            except ValueError:
                continue
            (tl, tr, br, bl) = bbox
            width_box = br[0]-tl[0]
            height_box = br[1]-tl[1]
            y = tl[1]+int(height_box/2)
            x = tl[0]+int(width_box/2)
            record.append((x,y,float(text)))
    #print(record)
    if len(record)>0:
        dtype = [('x', int), ('y', int), ('value', float)]
        a = np.array(record, dtype=dtype)
        a = np.sort(a, order='x')
        if stats.mode(a['y'])[1]>1:
            A_bool = [[math.isclose(i,j, abs_tol=max(int(height_box/2),5)) for i in a['y']] for j in a['y']]
            rows = [stats.mode(j)[0][0] for j in A_bool]
            a = a[rows]
    return a

def lin_approx(a):
    linear_model=np.polyfit(a['x'],a['value'],1)
    linear_model_fn=np.poly1d(linear_model)
    return linear_model_fn(a['x'])

# want error of approximation for every dot to be less than 5%
def big_rel_error(arr1, arr2, threshold = 0.05):
    err = np.absolute(1 - arr1/arr2) > threshold
    return err.any()

def drop_outlier(a):
    err = lin_approx(a) - a['value']
    a1 =  a[err**2 < max(err**2)]
    return a1

def well_approximated(a):
    a1=a
    while True:
        if len(a1)>2:
            #print('long enough')
            if big_rel_error(lin_approx(a1),a1['value']):
                a1 = drop_outlier(a1)
                #print('drop')
            else:
                confidence = 'confident'
                break
        else:
            confidence = 'unconfident'
            break
    return [a1, confidence]

def transform_units(units, X):
    if units == 'um':
        return X
    elif units == 'cm-1':
        return 10000/X
    elif units == 'eV':
        return 1.23984/X
    elif units == 'nm':
        return X/1000
    elif units == 'eV/h_bar':
        return 1.23984/X

def intersect(x1,v1,x2,v2):
    return v1-x1*(v2-v1)/(x2-x1)

def recalc(a1,x_box,cluster, xaxunits):
    pix1, _, val1 = a1[len(a1)//2-1]
    pix2, _, val2 = a1[len(a1)//2]
    starting_pix = x_box['xmin']
    X = np.array(cluster['coordinates'])[:,0]
    Y = np.array(cluster['coordinates'])[:,1]
    Y_rec = (max(Y) - Y)/(max(Y)-min(Y))
    #print('Y rec',type(Y_rec), Y_rec)
    X_rec = (val2-val1)/(pix2-pix1)*X + intersect(pix1+starting_pix,val1,pix2+starting_pix,val2)
    X_rec = transform_units(xaxunits,X_rec)
    #print('AFTER TRANSFORM X rec', type(X_rec), X_rec)
    cluster = np.array([(float(x),float(y)) for (x,y) in zip(X_rec,Y_rec)])
    return cluster[cluster[:, 0].argsort()].tolist()

def save_json(name, cluster_name, cluster, x_box, a1, xaxunits = 'um'):
    record = {}
    record['file_name']=name
    record['cluster_name'] = cluster_name[:-5]
    record['axes_units'] = {"x":"m*10-6","y":"1"}
    record['color'] = cluster['color']
    record['data'] = recalc(a1,x_box,cluster,xaxunits)
    #print(record)
    cluster_name_out = cluster_name[:-5]+'final_record.json'
    with open(os.path.join('images',cluster_name_out), 'w') as outfile:
        json.dump(record, outfile)