# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:24:24 2019

@author: Ashwin
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#Architecture
in_dim=4
hid_dim=100
out_dim=3


data=datasets.load_iris(return_X_y=False)
'''
data=pd.read_csv('seeds_dataset.csv',header=None)
data=np.array(data)
np.random.shuffle(data)
'''

#For data preprocessing
def to_categorical(y):
    new_y=np.zeros((len(y),out_dim))
    for n,i in enumerate(y):
        new_y[n][i]=1
    return new_y
    
X=np.array(data['data'])
y=np.array(data['target'],ndmin=0)

y=to_categorical(y)
np.random.seed(255)

d=np.hstack((X,y))
np.random.shuffle(d)
X,y=np.hsplit(d,np.array([4]))



#Functions requried for neural net
    
def activation_tanh(z):
            return(np.tanh(z))
            
def sigmoid(z):
    return(1/(1+np.exp(-z)))

def sigmoid_prime(z):
    return(sigmoid(z)*(1-sigmoid(z)))
    
def cost_fn(y,h_x,m):
    cost=(1/m)*np.sum((-y*np.log(h_x)-(1-y)*np.log(1-h_x)))
    
    #cost=0.5*np.sum((y-h_x)*(y-h_x))

    return cost


#CORE NN function
def learn(X,y,w1,w2,b1=0,b2=0,lr=0.01):
    '''Forward Propogation'''
    #dim of X=(m,4)
    a1=X
    
    #z2=(30,4) * (m,4).T
    z2=np.dot(w1,X.T)+b1
    #a2 dim =(30,m)#Chaing actibation here
    a2=sigmoid(z2)
    
    #(3,30)*(30,m)=>(3,m)
    z3=np.dot(w2,a2)+b2
    a3=sigmoid(z3)
    #(m,3)
    h_x=a3.T
        
    '''Back Propogate'''
    
    #Error at output_ (m,3)
    error_out=(h_x - y)
    #Gradient of prediction at output layer
    #(m,3)
    grad_out=sigmoid_prime(h_x)
    #Contribution of error in output layer(m,3)
    d_out=error_out * grad_out
   
    #(m,30)=(m,3)*(3,30)
    error_hid=np.dot(d_out,w2)
    grad_hid=sigmoid_prime(z2)#(30,m)
    #(m,30)=(m,30) * (30,m).T 
    d_hid=error_hid * grad_hid.T
       
    '''Weight Update'''
    #W=W+ dot(nextlayerError,Present layer Output)
    
    #(3,30)=(3,30)+((m,3)T . (30,m)T) 
    dw2=lr*np.dot(d_out.T,a2.T)
    #(30,4)=(30,4)+((m,30)T . (m,4))  
    dw1=lr*np.dot(d_hid.T,a1)
       
    cost=cost_fn(y,h_x,len(X))
    #print("COST==> ",cost)
    
    
    return cost,dw2,dw1
    

#Function to eval the predictions by the network compared to the actual
def predict(pred,y):
    score=0
    #print('I=',pred.shape)
    #print('J=',y.shape)
    for i,j in zip(pred.T,y.T):
        
        if np.argmax(i)==np.argmax(j):
            score+=1
    return(score)
    
def evaluate(w1,w2,X_test,y_test):
    a2=sigmoid(np.dot(w1,X_test.T))
    a3=sigmoid(np.dot(w2.T,a2))
    pred=a3
    #print(pred.shape)
    #print(y_test.shape)
    score=predict(pred,y_test)
    return score
    
#Util function to split data to train and test sets
def split_data(X,y,test_ratio=0.33):
    '''X_test,X_train,y_test,y_train'''
    

    test_len=int(len(X)*test_ratio)
    train_len=len(X)-test_len
    
    X_train=X[0:train_len]
    #print("X_train shape = ",X_train.shape)
    X_test=X[train_len:]
    y_train=y[0:train_len]
    y_test=y[train_len:]
    #print("y_test shape = ",y_test.shape)
    
    return X_test,X_train,y_test,y_train

    #Main Driver 
    
def NN(X,y,epoch=20,lr=0.01):
    c=[]
    X_test,X_train,y_test,y_train=split_data(X,y)
    w1=np.random.rand(hid_dim,in_dim)
    w2=np.random.rand(out_dim,hid_dim)
    b1=np.random.rand(hid_dim,1)
    b2=np.random.rand(out_dim,1)
    for i in range(epoch):
        cost,dw2,dw1=learn(X_train,y_train,w1,w2,b1,b2,lr)
        c.append(cost)
        if epoch%100==0:
            print("EPOCH=",i," COST==> ",cost)
        
        w2=w2-dw2
        w1=w1-dw1
            
    score=evaluate(X_test,y_test,w1,w2)
    print("Score=",score,'/',len(X_test))
    plt.plot(c)
    plt.show()
        
NN(X,y,1500,0.03)
