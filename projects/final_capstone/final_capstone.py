# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 23:33:48 2018

@author: konstantin
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline
import matplotlib.pyplot as plt

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
    
# Load the Credit Bank  dataset
ap_train=pd.read_csv("./data/application_train.csv")
ap_test=pd.read_csv("./data/application_test.csv")
 
# Success
print "Training dataset has {} data points with {} variables each.".format(*ap_train.shape)
print "Testing dataset has {} data points with {} variables each.".format(*ap_test.shape)


# head of the ap_train vector
ap_train.head()

ap_train.columns

# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
print(ap_train.info())

print(ap_train.isnull().sum().max())

ap_train['TARGET'].value_counts()

ap_train['TARGET'].astype(int).plot.hist()

# Missing values statistics
missing_values = missing_values_table(ap_train)
missing_values.head(20)

# Number of each type of column
ap_train.dtypes.value_counts()

# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in ap_train:
    if ap_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(ap_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(ap_train[col])
            # Transform both training and testing data
            ap_train[col] = le.transform(ap_train[col])
            ap_test[col] = le.transform(ap_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)


# one-hot encoding of categorical variables
ap_train = pd.get_dummies(ap_train)
ap_test = pd.get_dummies(ap_test)

print('Training Features shape: ', ap_train.shape)
print('Testing Features shape: ', ap_test.shape)

train_labels = ap_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
ap_train, ap_test = ap_train.align(ap_test, join = 'inner', axis = 1)

# Add the target back in
ap_train['TARGET'] = train_labels

print('Training Features shape: ', ap_train.shape)
print('Testing Features shape: ', ap_test.shape)


# Anomalies
(ap_train['DAYS_BIRTH'] / -365).describe()

(ap_train['DAYS_EMPLOYED']/ 365).describe()

# Create an anomalous flag column for train data
ap_train['DAYS_EMPLOYED_ANOM'] = ap_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
ap_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

ap_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');

# Create an anomalous flag column for testing data
ap_test['DAYS_EMPLOYED_ANOM'] = ap_test["DAYS_EMPLOYED"] == 365243
ap_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (ap_test["DAYS_EMPLOYED_ANOM"].sum(), len(ap_test)))

# Correlations

# Find correlations with the target and sort
correlations = ap_train.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

# Find the correlation of the positive days since birth and target
ap_train['DAYS_BIRTH'] = abs(ap_train['DAYS_BIRTH'])
ap_train['DAYS_BIRTH'].corr(ap_train['TARGET'])