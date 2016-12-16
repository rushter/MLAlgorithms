import numpy as np
import PIL
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import os
import glob
import argparse
import logging
import random
import tsne
from tsne import bh_sne
import images_plot
from scipy.spatial import ConvexHull




log = logging.getLogger()



#fig, ax = plt.subplots()

def tsne():
	"""
        This function makes the scatter graph using bhsne. The barnes hut stochastic neighborhood embedding algorithm is superior to regular tsne because bhsne scales. The barnes hutt performs the n-body simulation using oct trees in the three dimensional space. This makes the algorithm much faster and able to scale to a larger dataset. For large datasets bhsne is better.

        The link https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding describes tsne in detail. The parameter for the bh_sne are:
        perplexity
        theta

        The perplexity is 2 to the entropy of the probability distribution. It measures how many neighbors each data point will be connected to. When I raise the perplexity the images have more clusters.
        Theta measures the accuracy of the algorithm. It is the angle the data points are to each other. Large theta speeds up the algorithm but reduces the accuracy and small theta slows down the algorithm but increases the accuracy.
        """
        #read all the classfied files into a list
        #not only read the classified files into a list, but also keep it open for appending
	#randomarray=np.random.random(255, size=(1000, 784))
	randomarray=np.random.random((1000, 784))
	coordinates = bh_sne(randomarray, perplexity = 30, theta = .1) * 10
        print coordinates




if __name__ == '__main__':	
    streamhandler = logging.StreamHandler(sys.stdout)
    
    if args.logging_level==10:
       streamhandler.setLevel(logging.INFO)
       log.setLevel(logging.INFO)
    if args.logging_level==20:
       streamhandler.setLevel(logging.DEBUG)
       log.setLevel(logging.DEBUG)

    filehandler = logging.FileHandler("logging")
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    streamhandler.setFormatter(formatter)
    log.addHandler(streamhandler)
    tsne()
