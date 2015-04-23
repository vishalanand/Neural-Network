import numpy, argparse, os, re, random, math
import numpy as np
from collections import Counter

def sigmoid(x):
	#print "Hello",x
	return 1 / (1 + math.exp(-x))

def sigmoid1(X):
	return 1/(1+np.exp(-X*1.000))

def dsigmoid(X):
	return sigmoid(X)*(1-sigmoid(X))

print sigmoid(2);
print sigmoid1(2);