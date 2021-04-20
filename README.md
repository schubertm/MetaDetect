# What is MetaDetect?

MetaDetect is a post-processing tool for object detection neural networks. For each predicted Bounding Box, MetaDetect on one hand provides a method that predicts whether this particular Bounding Box intersects with the ground truth or not. This task can be understood as meta classifying between the two classes {IoU>=0.5} and {IoU<0.5} box-wise. On the other hand MetaDetect also provides a method for quantifying the uncertainty/confidence for each predicted Bounding Box by predicting IoU values via regression. 

# Preparation and evaluation


**1. Clone this file**

``` 
git clone https://github.com/schubertm/MetaDetect.git
cd MetaDetect/
```

**2. Packages and their version we used:**

* argparse
* glob
* imblearn==0.0
* json
* matplotlib==3.3.3
* numpy==1.17.3
* opencv-python==4.2.0.32
* os
* pandas==1.1.3
* scikit-learn==0.23.2
* sklearn==0.0
* smogn==0.1.2
* tensorflow==1.14.0
* tensorflow-gpu==1.14.0
* time
* tqdm==4.42.1
* xgboost==1.2.1

**3. Preparing data:**

Prepare a folder (e.g. "prediction") with a subfolder called "csv". Save the predictions **before Non-Maximum Suppression (NMS)** in the subfolder "csv" in the following way:

* save a .csv-file for every image (e.g. "image_name".csv) and save every prediction before NMS line by line with the following columns:
* [file_path	xmin	ymin	xmax	ymax	s	category_idx	prob_sum	prob_0	prob_1	prob_2	prob_3	prob_4	prob_5	prob_6	prob_7	dataset_box_id]
