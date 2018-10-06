# Machine Learning Engineer Nanodegree
## Project: Capstone Proposal and Capstone Project

### Definition

For the final project I’ve decided to join to one of competitions on Kaggle.com. 
	

Pneumonia accounts for over 15% of all deaths of children under 5 years old internationally. 
In 2015, 920,000 children under the age of 5 died from the disease. In the United States, 
pneumonia accounts for over 500,000 visits to emergency departments [1] and over 50,000 deaths in 2015 [2], 
keeping the ailment on the list of top 10 causes of death in the country.
While common, accurately diagnosing pneumonia is a tall order. It requires review of a chest radiograph (CXR) 
by highly trained specialists and confirmation through clinical history, vital signs and laboratory exams. 
Pneumonia usually manifests as an area or areas of increased opacity [3] on CXR. However, the diagnosis of 
pneumonia on CXR is complicated because of a number of other conditions in the lungs such as fluid overload (pulmonary edema),
bleeding, volume loss (atelectasis or collapse), lung cancer, or post-radiation or surgical changes. 
Outside of the lungs, fluid in the pleural space (pleural effusion) also appears as increased opacity on CXR. 
When available, comparison of CXRs of the patient taken at different time points and correlation with clinical 
symptoms and history are helpful in making the diagnosis.
CXRs are the most commonly performed diagnostic imaging study. A number of factors such as positioning of the 
patient and depth of inspiration can alter the appearance of the CXR [4], complicating interpretation further. 
In addition, clinicians are faced with reading high volumes of images every shift.
To improve the efficiency and reach of diagnostic services, the Radiological Society of North America (RSNA®) has 
reached out to Kaggle’s machine learning community and collaborated with the US National Institutes of Health, 
The Society of Thoracic Radiology, and MD.ai to develop a rich dataset for this challenge.


In this competition, I’m challenged to build an algorithm to detect a visual signal for pneumonia in medical images. 
Specifically, your algorithm needs to automatically locate lung opacities on chest radiographs.

	
### Data Description

   There are the following files in a dataset
   
•  **stage_1_train_images.zip** and **stage_1_test_images.zip** – training and test images.

•  **stage_1_train.csv** – training labels.
	
•  **stage_1_sample_submission.csv** - provides the IDs for the test set, as well as a 
    sample of what your submission should look like.

•  **stage_1_detailed_class_info.csv** - contains detailed information about the positive and 
   negative classes in the training set, and may be used to build more nuanced models.

### Data fields

•	**patientId**_ - A patientId. Each patientId corresponds to a unique image.

•	**x**_ - the upper-left x coordinate of the bounding box.

•	**y**_ - the upper-left y coordinate of the bounding box.

•	**width**_ - the width of the bounding box.

•	**height**_ - the height of the bounding box.

•	**Target**_ - the binary Target, indicating whether this sample has evidence of pneumonia.

All files can be downloaded from <https://www.kaggle.com/c/home-credit-default-risk/data>

### Solution statement

In the project I will be following the next process:

1. **Data overview**.

      • Example image visualizing.
  
      • Labels and data analysis.
	  
	  • Drawing  boxes.
	  
   
2. **Constructing a simple CNN with segmentation using Keras**.

     • Resample images.
	 
	 • Choosing network architecture.
	 
	 • Train and test the CNN.
	 
   
3. **Using transfer learning to get better accuracy**. 

     • Trying different CNN architectures like : VGG16, RESNET52, Xception, Mask R-CNN, AlexNet ect.
   
   
4. **Comparing the results and load the best on Kaggle**.

5. **Making a conclusion**.

 