import numpy as np
from data_utils import *
import random
from gradient_check import *

def initialize_parameters_nn(input_size,hidden_size,output_size):


    model={}
    
    # W1 is  parameter weight  for the first layer of neural network including bias term as the parameter. shape (hidden_size,input_size+1)
    model["W1"]=np.random.randn(hidden_size,input_size+1)*.001
    """ W2 is  parameter weight  for the second layer of neural network including bias term as the parameter. shape           (output_size,hidden_size  +1)"""
    model["W2"]=np.random.randn(output_size,hidden_size+1)*.001
    
    return model
    
    
    
def calculate_gradient_loss(X,model,y=None,reg=.20):
    
    #unpacked the parameters from nn model dictionary
    W1,W2=model["W1"],model["W2"]
  
    X=np.array([np.concatenate((np.array([1]),X))]).T
   
   
    loss=0.0
    Z1=W1.dot(X)
    
    A1=np.zeros_like(Z1)
    A1=sigmoid(Z1)
    
    A1=np.concatenate((np.array([[1]]),A1),0)
    Z2=W2.dot(A1)
    
    # MAXIMUM VALUE IS SUBSTRACTED FROM ALL
    Z2-=np.max(Z2)
    
    Z2=np.exp(Z2)
    
    Z2/=np.sum(Z2)
    
    #print Z2.shape
    
    loss=-1*np.log(Z2[y])
    
    loss=loss+0.5*reg*(np.sum(np.square(W1[:,1:]))+np.sum(np.square(W2[:,1:])))
    
    
    if(y==None):
       return loss
    
    grad={}
    
    dZ2=np.zeros_like(Z2)
    
    dZ2[:]=Z2
    dZ2[y]-=1

    dW2=dZ2.dot(A1.T)
    dA1=np.dot(W2.T,dZ2)
    #removing bias activation
    dA1=dA1[1:]   
    dZ1=dsigmoid(Z1)*dA1
    dW1=dZ1.dot(X.T)
    dx=np.dot(W1.T,dZ1) 
    dx=dx[1:]
    
    #ADDING REGULARIZATION TO WEIGHTS
    dW1[:,1:]+=reg*W1[:,1:]
    dW2[:,1:]+=reg*W2[:,1:]   
    
    
    grad["W1"]=dW1
    grad["W2"]=dW2
    grad["X"]=dx
    
    return loss,grad
    

def sigmoid(X):
    return 1/(1+np.exp(-X*1.000))
    

def dsigmoid(X):
     return sigmoid(X)*(1-sigmoid(X))    
    
    
Xtr,Ytr,Xte,Yte=load_CIFAR10('dataset/')#loaded Cifar10 data set as training set Xtr, labels of training set as Ytr, Xte of training set,Yte of Training set 

"""Converting Image data set to Raw Date Format"""
Xtr_rows=Xtr.reshape(Xtr.shape[0],Xtr.shape[1]*Xtr.shape[2]*Xtr.shape[3])
Xte_rows=Xte.reshape(Xte.shape[0],Xte.shape[1]*Xte.shape[2]*Xte.shape[3])

Xtr_rows=Xtr_rows-np.mean(Xtr_rows,0)
Xtr_rows=Xtr_rows/np.std(Xtr_rows,0)
    
model=initialize_parameters_nn(Xtr.shape[1]*Xtr.shape[2]*Xtr.shape[3],100,10)


print " normal ",calculate_gradient_loss(Xtr_rows[0],model,Ytr[0],.001)[1]["W2"][:3,:50]     

def grad_nn(W1):
    model["W2"]=W1
    return calculate_gradient_loss(Xtr_rows[0],model,Ytr[0],.001)[0]
    
    
print " analytical ",eval_numerical_gradient(grad_nn, model["W2"])  
   
