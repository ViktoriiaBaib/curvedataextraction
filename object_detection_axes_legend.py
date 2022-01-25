import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
from os import walk
import glob

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")

from utils import label_map_util_v2
from utils import visualization_utils as vis_util

from utils import ops as utils_ops

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.convert("RGB").getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')
            # Run inference
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def get_image_boxes_and_write_json(image_path,im_width, im_height,image, boxes, classes, scores, category_index, use_normalized_coordinates=True,max_boxes_to_draw=20,min_score_thresh=.5):
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box
            if use_normalized_coordinates:
                (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
            else:
                (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'NA'
            ending_string = '_'+str(class_name)+'_'+str(i)
            output_image_path = image_path[:-4]+ending_string+'.png'
            json_record = {}
            json_record['image_name'] = image_path
            json_record['image_size'] = []
            json_record['image_size'].append({'height':im_height, 'width':im_width})
            json_record['detected_box'] = []
            json_record['detected_box'].append({'box_image_name':output_image_path, 'label':str(class_name), 'score':int(100*scores[i]), 'xmin':left, 'xmax':right, 'ymin':top, 'ymax':bottom})
            subimage = Image.fromarray(image[top:bottom,left:right,:])
            subimage.save(output_image_path)
            output_json_path = output_image_path[:-4]+'.json'
            with open(output_json_path,'w') as outfile:
                json.dump(json_record,outfile)


if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# What model to download.
MODEL_NAME = 'inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_NAME,'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training','labelmap.pbtxt')

PATH_TO_SAMPLE_IMAGES_DIR = 'images'
SAMPLE_IMAGE_PATHS = []
for (dirpath, dirnames, filenames) in walk(PATH_TO_SAMPLE_IMAGES_DIR):
    print(filenames)
    SAMPLE_IMAGE_PATHS.extend([filename for filename in filenames if 'png' in filename])
    break
print(SAMPLE_IMAGE_PATHS)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util_v2.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

for ind, filename in enumerate(SAMPLE_IMAGE_PATHS):
    image_path = os.path.join(PATH_TO_SAMPLE_IMAGES_DIR,filename)
    image = Image.open(image_path)
    im_width, im_height = image.size
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    #Extraction and writing json
    get_image_boxes_and_write_json(image_path,im_width, im_height,image_np, output_dict['detection_boxes'],output_dict['detection_classes'],output_dict['detection_scores'], category_index, use_normalized_coordinates=True)

