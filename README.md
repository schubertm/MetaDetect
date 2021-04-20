# What is MetaDetect?

MetaDetect is a post-processing tool for object detection neural networks. For each predicted Bounding Box, MetaDetect on one hand provides a method that predicts whether this particular Bounding Box intersects with the ground truth or not. This task can be understood as meta classifying between the two classes {IoU>=0.5} and {IoU<0.5} box-wise. On the other hand MetaDetect also provides a method for quantifying the uncertainty/confidence for each predicted Bounding Box by predicting IoU values via regression. 

Arxiv: https://arxiv.org/pdf/2010.01695.pdf

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

1. Prepare a folder (e.g. "prediction") with a subfolder called "csv". Save the predictions **before Non-Maximum Suppression (NMS)** in the subfolder "csv" in the following way:

* save a .csv-file for every image (e.g. "image_name".csv) and save every prediction before NMS line by line with the following columns:
* \[file_path,	xmin,	ymin,	xmax,	ymax,	s,	category_idx,	prob_sum,	prob_0,	prob_1,	...,	prob_num_classes, dataset_box_id\]
 ![Bildschirmfoto_2021-04-20_14-46-32](https://user-images.githubusercontent.com/50663022/115398468-79eca000-a1e7-11eb-9846-200205a10944.png)

2. Prepare a folder (e.g. "ground_truth") with a subfolder called "csv". Save the ground truth boxes in the subfolder "csv" in the following way:

* save a .csv-file for every image (e.g. "image_name"\_gt.csv) and save every ground truth box line by line with the following columns:
* \[file_path, xmin,	ymin,	xmax,	ymax, category\_idx\]

**4. Preparing the scripts:**

* Go to configs/data_config.py and update default_df_path, default_gt_path, num_classes and CLASS_NAMES.

**5. Run the scripts:**

* Run the src/uq.py script:
``` 
cd src/
python3 uq.py
```
* Some evaluations can also be done:
``` 
cd ..
python3 plot_predictions.py
```
