import numpy as np
from data_utils import *
import random


def initialize_parameters_nn(input_size,hidden_size,output_size):


    model={} 
    model["W1"]=np.random.randn(hidden_size,input_size+1)*.001
    model["W2"]=np.random.randn(output_size,hidden_size+1)*.001
    #return model
    
    
    
def calculate_gradient_loss(X,model,y=None,reg=.10):
    #unpacked the parameters from nn model dictionary
    W1,W2=model["W1"],model["W2"]
  
    X=np.array([np.concatenate((np.array([1]),X))]).T
   
    loss=0.0
    Z1=W1.dot(X)
    
    A1=np.zeros_like(Z1)
    A1=sigmoid(Z1)
    
    A1=np.concatenate((np.array([[1]]),A1),0)
    Z2=W2.dot(A1)
    margin=Z2-Z2[y]+1
    margin[y]=0
    margin[margin<0]=0
    loss=np.sum(margin)
    
    if(y==None):
       return loss
    
    grad={}
    
    dZ2=np.zeros_like(Z2)
    
    dZ2[margin>0]=1
    dZ2[margin<0]=0
    dZ2[y]-=np.sum(margin>0)

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
    
      
    

