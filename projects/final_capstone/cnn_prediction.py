# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""

import pydicom
import cv2
import os
import numpy
import pandas as pd
from matplotlib  import  cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


Path_dicom_image = "all/stage_1_train_images"

Path_boxes_image = "all/stage_1_train_labels.csv"

path_additional_info = "all/stage_1_detailed_class_info.csv"

print(os.listdir("all"))

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(Path_dicom_image):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
            
boxes_labels = pd.read_csv(Path_boxes_image)
boxes_labels.info()
boxes_labels.head()   

additional_info = pd.read_csv(path_additional_info)
additional_info.info() 
additional_info.head()   
            
# Get ref file
patient_normal = pydicom.read_file(lstFilesDCM[0])

# Get image array
image_array_norm = patient_normal.pixel_array

# Plot image
plt.imshow(image_array_norm, cmap=plt.cm.bone) 


# Get ref file
patient_lung_op = pydicom.read_file(lstFilesDCM[8])

# Get image array
image_array_lung_op = patient_lung_op.pixel_array

# Create figure and axes
fig,ax = plt.subplots(1)
# Plot image
ax.imshow(image_array_lung_op, cmap=plt.cm.bone) 

print(additional_info.iloc[8])
print(boxes_labels.iloc[8])


# Create a Rectangle patch
X_width = boxes_labels['width'].iloc[8]
Y_width = boxes_labels['height'].iloc[8]

# bottom left coordinates for rect = upper left X of dicom
X = boxes_labels['x'].iloc[8]

# bottom left coordinates  rectange Y = Ydicom + y_widht
Y = boxes_labels['y'].iloc[8] + Y_width

box = patches.Rectangle((X,Y),X_width,Y_width,linewidth=1,edgecolor='r',facecolor='none')

# Add the box to the Plot
ax.add_patch(box)
plt.show()
