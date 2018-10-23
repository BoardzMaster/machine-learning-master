
#import libs
import time

import pydicom
import cv2
import os
import sys
import numpy
import pandas as pd
from matplotlib  import  cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


# Root directory of the RCNN lib
ROOT_DIR = os.path.abspath("/Mask_RCNN-master/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the lib
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Import RCNN override class
from Mask_RCNN_model import *


Path_train_dicom_image = "all/stage_1_train_images"

Path_train_boxes_image = "all/stage_1_train_labels.csv"

path_additional_info = "all/stage_1_detailed_class_info.csv"

print(os.listdir("all"))

def dataset_load (image_dir, image_info_file):
    lstFilesDCM = []
    for dirName, subdirList, fileList in os.walk(image_dir):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    image_info = {pat: [] for pat in lstFilesDCM}
    for index, row in image_info_file.iterrows():
        pat = os.path.join(image_dir, row['patientId']+'.dcm')
        image_info[pat].append(row)
    return lstFilesDCM, image_info
            
boxes_labels = pd.read_csv(Path_train_boxes_image)
boxes_labels.info()
boxes_labels.head()   

additional_info = pd.read_csv(path_additional_info)
additional_info.info() 
additional_info.head()

image_file_list, image_info = dataset_load(Path_train_dicom_image, boxes_labels )

ORIG_SIZE = 1024

dataset_train = DetectorDataset(image_file_list, image_info, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()


# Load and display random sample and their bounding boxes

class_ids = [0]
while class_ids[0] == 0:  ## look for a mask
    image_id = random.choice(dataset_train.image_ids)
    image_fp = dataset_train.image_reference(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)

print(image.shape)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
    masked += image[:, :, 0] * mask[:, :, i]
plt.imshow(masked, cmap='gray')
plt.axis('off')

print(image_fp)
print(class_ids)


COCO_WEIGHTS_PATH = "/Mask_RCNN-master/mask_rcnn_coco.h5"
model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

# Exclude the last layers because they require a matching
# number of classes
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])

LEARNING_RATE = 0.006

# Train Mask-RCNN Model
import warnings
warnings.filterwarnings("ignore")



model.train(dataset_train,
            learning_rate=LEARNING_RATE,
            epochs=6,
            layers='all')

history = model.keras_model.history.history
for k in history: history[k] = history[k] + history[k]

# # Get ref file
# patient_normal = pydicom.read_file(image_file_list[0])


# # Get image array
# image_array_norm = patient_normal.pixel_array

# # Plot image
# plt.imshow(image_array_norm, cmap=plt.cm.bone) 


# # Get ref file
# patient_lung_op = pydicom.read_file(image_file_list[8])

# # Get image array
# image_array_lung_op = patient_lung_op.pixel_array

# # Create figure and axes
# fig,ax = plt.subplots(1)
# # Plot image
# ax.imshow(image_array_lung_op, cmap=plt.cm.bone) 

# print(additional_info.iloc[8])
# print(boxes_labels.iloc[8])


# # Create a Rectangle patch
# X_width = boxes_labels['width'].iloc[8]
# Y_width = boxes_labels['height'].iloc[8]

# # bottom left coordinates for rect = upper left X of dicom
# X = boxes_labels['x'].iloc[8]

# # bottom left coordinates  rectange Y = Ydicom + y_widht
# Y = boxes_labels['y'].iloc[8] + Y_width

# box = patches.Rectangle((X,Y),X_width,Y_width,linewidth=1,edgecolor='r',facecolor='none')

# # Add the box to the Plot
# ax.add_patch(box)
# plt.show()
