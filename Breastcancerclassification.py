# Import dependencies.
'''
Numpy: Library used to create numpy arrays.
Pandas: Library used to create Panda dataframe.
sklearn.datasets: From here we import breast cancer Data.
LogisticRegression: It is used to detect binary values.(Cancer cells / not)
accuracy_score: Checks the number of correct predictions our model is making.
'''

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




# Data Collection and Processing
# Loading data from Sklearn
breast_cancer_datasets = sklearn.datasets.load_breast_cancer()
print(breast_cancer_datasets)



# Load data to panda dataframe.
data_frame = pd.DataFrame(breast_cancer_datasets.data , columns = breast_cancer_datasets.feature_names)

# Print first 5 rows of dataframe
print(data_frame.head())






# Adding the 'target' column to the data frame.
data_frame['label'] = breast_cancer_datasets.target

# print last 5 rows of the dataframe.
print(data_frame.tail())






# Number of rows and columns in the datasets. {No. of rows , No. of column}
print(data_frame.shape)

# Getting some info about data.
print(data_frame.info())

# Checking for missing values.
print(data_frame.isnull().sum())

# Statisticals measures about the datasets.
print(data_frame.describe())

# Checking the distribution of Target Variables (1:Benign , 0:Malignant)
print(data_frame['label'].value_counts())

# All values for Malignant and Benign are taken and then we are returning mean of this two.
print(data_frame.groupby('label').mean())





# Separating the features and target .{Columns: axis=1, Rows: axis=0} 
x=data_frame.drop(columns='label',axis=1)
y=data_frame['label']
print(x,y)





# Splitting Data to Training and Testing data.
# x_train: training data
# x_test: testing data.
# y_train: stores corresponding labels of x_train. 
# y_test: stores corresponding labels of x_test.
# we basically keep 20% of Testing data and 80% of Training data. So for keeping the testing data to be 20% we have mentioned test_size=0.2.
# random_state is a way splitting data. 

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=2)
print(x.shape,x_train.shape,x_test.shape)





# Model Training
model = LogisticRegression()

# Training Logistic Regression Model using Training data.
# model.fit(): fits the dataset to our model.After running these function our model gets trained.
model.fit(x_train,y_train)






# Model Evaluation
# Accuracy Score

# Accuracy on training data.
x_train_prediction = model.predict(x_train) 
train_data_accuracy = accuracy_score(y_train,x_train_prediction)
print("Accuracy on our Training data: ",train_data_accuracy)

# Accuracy on test data.
x_test_prediction = model.predict(x_test) 
test_data_accuracy = accuracy_score(y_test,x_test_prediction)
print("Accuracy on our Training data: ",test_data_accuracy)



# Building a Predictive System.
input_data = (9.504,12.44,60.34,273.9,0.1024,0.06492,0.02956,0.02076,0.1815,0.06905,0.2773,0.9768,1.909,15.7,0.009606,0.01432,0.01985,0.01421,0.02027,0.002968,10.23,15.66,65.13,314.9,0.1324,0.1148,0.08867,0.06227,0.245,0.07773)
# Changing input data to numpy array.
input_data_numpy_arrays = np.asarray(input_data)

# Reshape the numpy array as we are predicting for one datapoint.
input_data_reshape = input_data_numpy_arrays.reshape(1,-1)
prediction  = model.predict(input_data_reshape)

if(prediction[0]==1):
    print("Breast Cancer is Benign type")
else:
    print("Breast Cancer is Malignant type")
