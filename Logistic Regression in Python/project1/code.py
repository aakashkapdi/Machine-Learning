

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Data
data=pd.read_csv("Social_Network_Ads.csv")
x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

#Splitting data for test and traing the hypothesis
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Fitting Logistic Regression to training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

#Prediction
y_pred=classifier.predict(x_test)

#Performance of the Model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Visualisation
#from matplotlib.colors import ListedColormap
x_set=x_train;
x_axis,y_axis=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                          np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))                       
x_for_graph=np.array([x_axis.ravel(),y_axis.ravel()])
color_train_pts=np.where(classifier.predict(x_set)==1,'blue','k')
color_all_pts=np.where(classifier.predict(x_for_graph.T)==1,'g','r')
plt.scatter(x_axis,y_axis,c=color_all_pts)
plt.scatter(x_set[:,0],x_set[:,1],c=color_train_pts)
plt.show()





