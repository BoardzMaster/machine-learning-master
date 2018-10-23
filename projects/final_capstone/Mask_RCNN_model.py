import sys
import os
import numpy as np
import pydicom
import cv2

# Root directory of the RCNN lib
ROOT_DIR = os.path.abspath("/Mask_RCNN-master/")

# Mask RCNN lib
sys.path.append(ROOT_DIR)  # To find local version of the lib
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


class DetectorConfig(Config):
    """Override config class in mrcnn lib for Pneumonia detection on the RSNA kaggle dataset.
    """

    # Give the configuration a recognizable name
    NAME = 'Pneumonia'

    # Train on 1 GPU and 1 images per GPU. We can put multiple images on each

    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    BACKBONE = 'resnet50'

    NUM_CLASSES = 2  # background (BG) + 1 Pneumonia classes

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.78  ## match target distribution
    DETECTION_NMS_THRESHOLD = 0.01

    STEPS_PER_EPOCH = 50

config = DetectorConfig()
config.display()


class DetectorDataset(utils.Dataset):
    """Override Dataset class in mrcnn lib for Pneumonia detection on the RSNA kaggle dataset.
       Could be used for loading train/test/validation datasets
    """

    def __init__(self, image_list, image_information, height, width):
        super().__init__(self)

        # Add classes - we need just 1 class ('Pneumonia')
        self.add_class('Pneumonia', 1, 'Lung Opacity')

        # add images
        for i, file_path in enumerate(image_list):
            image_info_box = image_information[file_path]
            self.add_image('Pneumonia', image_id=i, path=file_path,
                           image_information=image_info_box, image_height=height, image_width=width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        file_path = info['path']
        dicom_file = pydicom.read_file(file_path)
        image = dicom_file.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        image_information = info['image_information']

        box_counter = len(image_information)
        
        if box_counter == 0:
            mask = np.zeros((info['image_height'], info['image_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['image_height'], info['image_width'], box_counter), dtype=np.uint8)
            class_ids = np.zeros((box_counter,), dtype=np.int32)
            for i, box in enumerate(image_information):
                if box['Target'] == 1:
                    x = int(box['x'])
                    y = int(box['y'])
                    width = int(box['width'])
                    height = int(box['height'])
                    
                    mask_inst = mask[:, :, i].copy()

                    cv2.rectangle(mask_inst, (x, y), (x + width, y + height), 255, -1)
                    mask[:, :, i] = mask_inst
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)

model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)