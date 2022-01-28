*Automated curve data extraction. Works with colored curves. Cross-platform. Color decomposition with DCD. X units parser. Automatic axes scale recognition.*

1. Download and unzip **folder** from https://drive.google.com/file/d/1e0UTKwhgJN9DuD2qYsLcWcKd6WomvRkl/view?usp=sharing

2. Download scripts above.

### Part I: install packages and TF2 object detection API

3. Create anaconda variable
```
conda create -n imgrec python=3.9
conda install -n imgrec pip
conda activate imgrec
```
4. Install Tensorflow 2.7
```
pip install TensorFlow==2.7
```
5. Install protobuf (probably it is already there after previous step; try installation via brew for macOS)
```
pip install protobuf
```
6. Install some packages
```
pip install easyocr
pip install opencv-python==4.5.4.60
pip install dominant-color-detection
```
7. Install TF object detection API

Create a directory for Tensorflow Models repository. Go to this directory and do
```
git clone https://github.com/tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```
8. Check installation
```
python object_detection/builders/model_builder_tf2_test.py
```

### Part II: use pre-trained model

9. 
```
cd object_detection
mkdir inference_graph
```
10. copy the content of the inference_graph from **folder** to the corresponding directory

11.
``` 
mkdir training 
```
copy labelmap.pbtxt from **folder** to training/.

12. 
```
mkdir images
```
copy file 0.png from **folder** to images/.

13. copy file label_map_util_v2.py from **folder** to utils/.

14. copy script object_detection_axes_legend.py from above (step 2) to  models/research/object_detection/.

15. run this file from models/research/object_detection/ as
```
python object_detection_axes_legend.py
```
You should obtain json and png files for Legend, X axis and Y axis in images/

### Part III: run the rest of the project

16. copy scripts posterization.py and color_decomposition.py from above (step 2) to models/research/object_detection/.

17. Run file color_decomposition.py as
```
python color_decomposition.py
```
You should obtain png of color cut and palette as well as json and png of separate clusters in images/

18. Copy scripts final-record.py and final_record_func.py from above (step 2) to models/research/object_detection/.

19. Run final-record.py as
```
python final-record.py 
```
You can also specify x axis units running as
```
python final-record.py -x um
```
You should obtain json files of final records for all successfully detected clusters.

To work with your figure put figure.PNG to images/.

*Regarding the TFOD API part, thanks to https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model*
