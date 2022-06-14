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
def get_scaling(axis_json, axis_img, reader, allowlist, axis):
    box = axis_json['detected_box'][0]
    width = box['xmax']-box['xmin']
    height = box['ymax']-box['ymin']
    axis_img = cv2.copyMakeBorder(axis_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT,value=(255,255,255))
    results = reader.readtext(axis_img, min_size = 10, mag_ratio=2, allowlist=allowlist)
    a = clean_array(results, axis)
    [a1, confidence] = well_approximated(a, axis)
    return [confidence, a1, box]

def clean_array(results, axis):
    a = np.array([])
    record = []
    m_height_box = 10
    m_width_box = 5
    for (bbox, text, prob) in results:
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
            m_height_box = min(height_box, 10)
            m_width_box = min(width_box,5)
    #print(record)
    if len(record)>0:
        dtype = [('x', int), ('y', int), ('value', float)]
        a = np.array(record, dtype=dtype)
        a = np.sort(a, order=axis)
        axis2 = 'xy'.replace(axis,'')
        param_box = {'x':max(int(m_height_box/2),6), 'y':max(int(m_width_box/2),3)}
        if stats.mode(a[axis2])[1]>1:
            A_bool = [[math.isclose(i,j, abs_tol=param_box[axis]) for i in a[axis2]] for j in a[axis2]]
            rows = [stats.mode(j)[0][0] for j in A_bool]
            a = a[rows]
    return a

def lin_approx(a,axis):
    linear_model=np.polyfit(a[axis],a['value'],1)
    linear_model_fn=np.poly1d(linear_model)
    return linear_model_fn(a[axis])

# want error of approximation for every dot to be less than 5%
def big_rel_error(arr1, arr2, threshold = 0.05):
    err = np.absolute(1 - arr1/arr2) > threshold
    return err.any()

def drop_outlier(a,axis):
    err = lin_approx(a,axis) - a['value']
    a1 =  a[err**2 < max(err**2)]
    return a1

def well_approximated(a, axis):
    a1=a
    while True:
        if len(a1)>2:
            #print('long enough')
            if big_rel_error(lin_approx(a1,axis),a1['value']):
                a1 = drop_outlier(a1,axis)
                #print('drop')
            else:
                confidence = 'confident'
                break
        else:
            confidence = 'unconfident'
            break
    return [a1, confidence]

def intersect(x1,v1,x2,v2):
    return v1-x1*(v2-v1)/(x2-x1)

def axis_rec(axis_arr, a1, box, axis):
    pix1 = a1[len(a1)//2-1][axis]
    val1 = a1[len(a1)//2-1]['value']
    pix2 = a1[len(a1)//2][axis]
    val2 = a1[len(a1)//2]['value']
    starting_pix = box[axis+'min']
    return (val2-val1)/(pix2-pix1)*axis_arr + intersect(pix1+starting_pix,val1,pix2+starting_pix,val2)

def recalc(x_a1,x_box,y_a1,y_box,cluster):
    X = np.array(cluster['coordinates'])[:,0]
    Y = np.array(cluster['coordinates'])[:,1]
    X_rec = axis_rec(X, x_a1, x_box, 'x')
    #print('X data type ', type(X_rec))
    Y_rec = axis_rec(Y, y_a1, y_box, 'y')
    #print('Y data type ', type(Y_rec))
    cluster = np.array([(float(x),float(y)) for (x,y) in zip(X_rec,Y_rec)])
    result = cluster[cluster[:, 0].argsort()].tolist()
    #print('result type', type(result))
    return result

def save_json(name, cluster_name, cluster, x_box, x_a1, y_box, y_a1):
    record = {}
    record['file_name']=name
    record['cluster_name'] = cluster_name[:-5]
    record['axes_units'] = "na"
    record['color'] = cluster['color']
    record['data'] = recalc(x_a1,x_box,y_a1,y_box,cluster)
    #print(record)
    cluster_name_out = cluster_name[:-5]+'final_record.json'
    with open(os.path.join('images',cluster_name_out), 'w') as outfile:
        json.dump(record, outfile)
    X = np.array(record['data'])[:,0]
    Y = np.array(record['data'])[:,1]
    fig = plt.figure(figsize=(8,6))
    plt.plot(X,Y,color = record['color'])
    fig.savefig(os.path.join('images',cluster_name_out)[:-4]+'png')
    plt.close('all')
