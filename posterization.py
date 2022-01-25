from __future__ import division
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json
import cv2
from PIL import Image, ImageDraw
from PIL.Image import Image as Image_type

def read_image(image):
    if type(image) is not Image_type:
        img = Image.open(image)
    else:
        img = image
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    im = np.asarray(img)
    im = im[:, :, :3]
    return im

def remove_legend(img, legend_filename):
    """given image and path to legend json record; open json record, find box with coordinates, return image without legend + 5 pixels padding"""

    f = open(legend_filename)
    data = json.load(f)
    legend_box = data['detected_box'][0]

    return cv2.rectangle(img, (legend_box['xmin']-5,legend_box['ymin']-5), (legend_box['xmax']+5,legend_box['ymax']+5), (255, 255, 255), -1)

def preprocess(image_filename, legend_filenames):
    """cuts legend and all b/w objects"""
    img0 = read_image(image_filename)
    hsv0 = cv2.cvtColor(img0,cv2.COLOR_RGB2HSV)
    hsv = hsv0
    for i in range(len(hsv0)):
        for j in range(len(hsv0[0])):
            if hsv0[i,j,1] < 124 or hsv0[i,j,2] < 124:
                hsv[i,j] = [0, 0 , 255]
    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    if len(legend_filenames)>0:
        for legend_filename in legend_filenames:
            img = remove_legend(img, legend_filename)
    return img

def hex2rgb(hex_code):
    h = hex_code.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

def squared_distance(p, q):
    """given points p and q returns the sum of the squares"""   
    return sum((p_i - q_i) ** 2 for p_i, q_i in zip(p, q))

def closest_index(pixels,  colors):
    """given a point a list of points, returns the index of the closest element"""
    return min(range(len(colors)), key=lambda i: squared_distance(pixels, colors[i]))

def get_matrix(image_array, colorsrgb):
    im = image_array[:, :, :3]/255
    return [[closest_index(pixel, colorsrgb) for pixel in row] for row in im]

def get_cluster_data(matrix,index):
    """returns list of (x,y) pixel coordinates of pixels associated with this cluster"""
    return [(j,i) for i,_ in enumerate(matrix) for j,_ in enumerate(matrix[i]) if matrix[i][j]==index]

def save_cluster(cluster, index, colorhex, img_filename):
    """save cluster image"""

    fig = plt.figure()
    plt.scatter(np.array(cluster)[:,0],np.array(cluster)[:,1],s=0.01,c=colorhex)
    plt.gca().invert_yaxis()
    plt.axis('off')
    fig.savefig(img_filename[:-4]+'_colorcut_cluster_'+str(index)+'.png')
    plt.cla()
    plt.clf()
    plt.close()

def save_json(cluster,color,index,img_filename):
    """write (x,y) in coordinates, palette colors"""
    record = {}
    record['color'] = color
    record['coordinates'] = cluster
    with open(img_filename[:-4]+'_colorcut_cluster_'+str(index)+'.json', 'w') as outfile:
        json.dump(record, outfile)

def save_palette(pixels, img_filename):
    """save the pixels"""
    plt.imsave(arr=np.array(pixels), fname=img_filename[:-4]+'_colorcut_palette.png')
    plt.cla()
    plt.clf()
    plt.close()

def ygroups(ys):
    ys_ref = range(ys[0],ys[0]+len(ys))
    if sum(np.array(ys)-ys_ref) == 0:
        ys_res = int(np.mean(ys))
        comment = ''
    else:
        dif = list(np.array(ys)-ys_ref)
        classes = list(set(dif))
        ys_res = [int(np.mean([y for i,y in enumerate(ys) if dif[i]==cl])) for cl in classes]
        comment = 'multiple'
    return [ys_res, comment]

def data_structured(cluster):
    structured = []
    Xs = np.array(cluster)[:,0]
    for ind in range(min(Xs),max(Xs)+1):
        ys = sorted([item[1] for item in cluster if item[0]==ind])
        comment = ''
        if len(ys)==0:
            comment = 'gap'
            ys_res = []
        else:
            [ys_res,comment] = ygroups(ys)
        structured.append({'x': ind, 'y': ys_res, 'c': comment})
    return structured

def data_score_mult(cluster):
    structured = data_structured(cluster)
    mults = len([entry for entry in structured if entry['c']=='multiple'])
    gaps = len([entry for entry in structured if entry['c']=='gap'])
    tot = len(structured)
    return 1-mults/(tot-gaps)
