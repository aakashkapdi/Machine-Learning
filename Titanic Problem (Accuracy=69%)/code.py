#importing packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
raw_data=pd.read_csv('train.csv')
raw_test_data=pd.read_csv('test.csv')

#testing for missing values
print("Null Values in Training Data=",raw_data.isna().any())
print(raw_data.isna().sum())
print("Null Values In Test Data:=",raw_test_data.isna().any())
print(raw_test_data.isna().sum())

#filling 2 missing values in Fare Coloumn for training and test data
raw_data['Fare'].fillna(raw_data[['Fare']].mean(), inplace=True)
raw_test_data['Fare'].fillna(raw_test_data['Fare'].mean(), inplace=True)

#testing for filled in Values
print("Null Values in Training Data=",raw_data.isna().any())
print(raw_data.isna().sum())
print("Null Values In Test Data:=",raw_test_data.isna().any())
print(raw_test_data.isna().sum())


#VISUALISATION OF DATA

color_pts=np.where(raw_data['Survived']==1,'g','r')

#No of Relatives v/s Survival
plt.scatter(raw_data['SibSp'],raw_data['Parch'],c=color_pts)
plt.xlabel('No of passengers whose Siblings/ Spouses on board')
plt.ylabel('No of passengers whose Parents/Children on board')
plt.title('No Of Relatives vs Survival')
plt.show()

#Fare  and Ticket Class vs Survival
plt.scatter(raw_data['Pclass'],raw_data['Fare'],c=color_pts)
plt.xlabel('Ticket Class')
plt.ylabel('Fare')
plt.title('Fare and Ticket Class vs Survival')
plt.show()

#Filtering Dataset For Algorithm
x_train=raw_data.iloc[:,[2,6,7,9]]
y_train=raw_data.iloc[:,1]
x_test=raw_test_data.iloc[:,[1,5,6,8]]


#splitting data for training and testing
from sklearn.model_selection import train_test_split
x_train_split,x_test_split,y_train_split,y_test_split=train_test_split(x_train,y_train,test_size=0.25,random_state=0)

#Implementing Support Vector Machine

from sklearn import svm
clf=svm.SVC(gamma=0.001,C=100,kernel='rbf')
clf.fit(x_train_split,y_train_split)

#Prediction
print("Prediction on training (test_split ) set=",clf.predict(x_test_split));
print("Prediction on test set=",clf.predict(x_test));
      
#Accuracy of the Modal
print("Accuracy on Training Set=",clf.score(x_train_split,y_train_split)*100);
print("Accuracy on Test Set=",clf.score(x_test_split,y_test_split)*100);


#writing on CSV file
predictions=clf.predict(x_test)
np.savetxt("submission.csv", predictions, delimiter=" ")




