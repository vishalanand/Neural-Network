import os,sys,csv,copy,random,itertools
from operator import itemgetter
from collections import defaultdict

import numpy as np
import scipy
import scipy.spatial.distance
from numpy.linalg import svd

from tsne import tsne
import matplotlib.pyplot as plt

def tsne_viz(mat=None, rownames=None, indices=None, colors=None, output_filename=None, figheight=40, figwidth=50, display_progress=False): 
    """2d plot of mat using tsne, with the points labeled by rownames, 
    aligned with colors (defaults to all black).
    If indices is a list of indices into mat and rownames, 
    then it determines a subspace of mat and rownames to display.
    Give output_filename a string argument to save the image to disk.
    figheight and figwidth set the figure dimensions.
    display_progress=True shows the information that the tsne method prints out."""
    if not colors:
        colors = ['black' for i in range(len(rownames))]
    temp = sys.stdout
    if not display_progress:
        # Redirect stdout so that tsne doesn't fill the screen with its iteration info:
        f = open(os.devnull, 'w')
        sys.stdout = f
    tsnemat = tsne(mat)
    sys.stdout = temp
    # Plot coordinates:
    if not indices:
        indices = range(len(rownames))        
    vocab = np.array(rownames)[indices]
    xvals = tsnemat[indices, 0] 
    yvals = tsnemat[indices, 1]
    # Plotting:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(40)
    fig.set_figwidth(50)
    ax.plot(xvals, yvals, marker='', linestyle='')
    # Text labels:
    for word, x, y, color in zip(vocab, xvals, yvals, colors):
        ax.annotate(word, (x, y), fontsize=8, color=color)
    # Output:
    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
    else:
        plt.show()

def get_usefull_word_vec(source_file):
  fp1=open(source_file,'r')
  count=0;
  word_vector=[]
  word_name=[]
  for line in fp1:
    if count ==0:
       print line
       count=1
    else:
        text=line.split()
        temp_vector=[float(x) for x in text[1:]]
        count+=1
        word_vector.append(temp_vector)
        word_name.append(text[0])

  count-=1
  print count
  fp1.close()
  return word_vector,word_name

word_vec,word_name=get_usefull_word_vec('vectors_50.txt')
print len(word_vec)
print len(word_name)
print word_name[:5]
print 'tsne work started '  
tsne_viz(mat=np.array(word_vec[:1000]), rownames=word_name[:1000])
