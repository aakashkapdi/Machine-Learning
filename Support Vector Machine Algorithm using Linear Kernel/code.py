#Importing Libraries


from sklearn import datasets
from sklearn import svm

#loading dataset
digits=datasets.load_digits()
x,y=digits.data,digits.target

#splitting data for testing and training
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=0)

#Classifier
clf=svm.SVC(gamma=0.001,C=100,kernel='linear')
clf.fit(x_train,y_train)


#Prediction
print("Prediction=",clf.predict(x_test));

#Accuracy of the Modal
print("Accuracy on Training Set=",clf.score(x_train,y_train)*100);
print("Accuracy on Test Set=",clf.score(x_test,y_test)*100);

