*Automated curve data extraction. Works with colored curves. Cross-platform. Color decomposition with DCD. X units parser. Automatic axes scale recognition.*

Download and unzip **folder** from https://drive.google.com/file/d/1e0UTKwhgJN9DuD2qYsLcWcKd6WomvRkl/view?usp=sharing
Download scripts above.

### Part I: install TF2 object detection API

1. Create anaconda variable
```
conda create -n imgrec
conda activate imgrec
which python
which pip
```
2. Install Tensorflow 2.7.0
```
pip install TensorFlow=2.7.0
```
3. Install protobuf (via brew for macOS)

4. Install TF object detection API

Create Tensorflow Models repository. Move there and do
```
git clone https://github.com/tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```
5. Check installation
```
python object_detection/builders/model_builder_tf2_test.py
```

### Part II: use pre-trained model

6. 
```
cd object_detection
mkdir inference_graph
```
7. copy the content of the inference_graph from **folder** to the corresponding directory

8.
``` 
mkdir training 
```
copy labelmap.pbtxt from **folder** to training/.

9. 
```
mkdir images
```
copy file 0.png from **folder** to images/.

10. copy file label_map_util_v2.py from **folder** to utils/.

11. copy script object_detection_axes_legend.py from above to  models/research/object_detection/.

13. run this file from models/research/object_detection/ as
```
python object_detection_axes_legend.py
```
You should obtain json and png files for Legend, X axis and Y axis in images/

### Part III: run the rest of the project

14. copy scripts posterization.py and color_decomposition.py from above to models/research/object_detection/.

15. Run file posterization.py as
```
python posterization.py
```
You should obtain png of color cut and palette as well as json and png of separate clusters in images/

16. Copy scripts final_record.py and final_record_func.py from above to models/research/object_detection/.

17. Run final_record.py as
```
python final_record.py 
```
You can also specify x axis units running as
```
python final_record.py -x um
```
You should obtain json files of final records for all successfully detected clusters.

To work with your figure put figure.PNG to images/.

*Regarding the TFOD API part, thanks to https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model*