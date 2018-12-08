import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_data(x,y):
    plt.plot(x, y,"r+")
    plt.show();

def cost_function(x,y,theta_0,theta_1,m):
    hypothesis=theta_0+theta_1*x;
    squared_error=(hypothesis-y)**2;
    j=(1/(2*m))*sum(squared_error)
    return j;

def gradiant_descent(x,y) :
    theta_0=0;
    theta_1=0;
    iterations=20;
    learning_rate=0.0001;
    m=len(x);
    for i in range(iterations) :
        hypothesis=theta_0+theta_1*x;
        error=hypothesis-y;
        der_cost_function_0=(1/m)*sum(error);
        der_cost_function_1=(1/m)*sum(error*x);
        theta_0=theta_0-(learning_rate*der_cost_function_0);
        theta_1=theta_1-(learning_rate*der_cost_function_1)
        print(theta_0,theta_1,cost_function(x,y,theta_0,theta_1,m));
        plt.plot(i,cost_function(x,y,theta_0,theta_1,m),"r+")
    plt.show();
    return theta_0,theta_1;

def plot_hypothesis(theta_0,theta_1):
    plt.plot(x,y,"r+")
    for i in range(0,100) :
        hypothesis=theta_0+(theta_1*i);
        plt.plot(i,hypothesis,"b^")
    plt.show();



data=pd.read_csv('train.csv');
x=data.iloc[:,0]
y=data.iloc[:,1]
x=np.array(x)
y=np.array(y)
plot_data(x,y);
[theta_0,theta_1]=gradiant_descent(x,y);
plot_hypothesis(theta_0,theta_1);