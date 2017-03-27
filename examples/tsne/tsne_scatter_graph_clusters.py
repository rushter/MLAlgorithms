import numpy as np
import PIL
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import os
import glob
from PIL import Image
import argparse
import logging
from PIL import Image,ImageOps,ImageDraw
import random
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.path import Path
import png
import tsne
from tsne import bh_sne
import images_plot
#import read_points_and_draw_convex_hull
from scipy.spatial import ConvexHull




log = logging.getLogger()
classifier_dict={}
convex_hull_points = []
classifier_is = None
reverse_classifier={}
classified_files = {}
contourfiles = []
coordinates = []

def onclick(event):
	"""
        This onclick works in concert with the tkAgg package to detect clicks. 
        The key detection is necessary for this program to classify images. The clicks around the cluster make the convex hull. See attached pictures.
        """
        global classifier_is
        global convex_hull_points
#	f.write('xdata=%f, ydata=%f\n' %
#          (event.xdata, event.ydata))
	print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
        #classifier_dict[classifier_is].append((event.xdata,event.ydata))
	#print convex_hull_points
        convex_hull_points.append([event.xdata,event.ydata])
        #classifier_dict[classifier_is+'<>'+__repr__(event.xdata))+'<>'+__repr__(event.ydata))] = #contour_file_name

def onkeypress(event):
	"""
        This function records the key pressed. When the escape key is pressed then the convex hull is recorded. The first key that is pressed following the recording of the convex hull signifies the class of the images within the hull.
        """
        global classifier_is
        global convex_hull_points
	global coordinates 
	global classified_files_list       
	global files_classifier_list       
        new_classified_files=[]
        new_files_classifiers=[]
        if event.key == 'backspace':
            for ij,l in enumerate(classified_files_list):
		print ("classifier is = ",classifier_is, " and files_classifier_list = ",files_classifier_list[ij])
                if files_classifier_list[ij]==classifier_is:
                       continue
                new_classified_files.append(l)
                new_files_classifiers.append(files_classifier_list[ij])
            ff = open('classified_files.txt','w')
            for ij,n in enumerate(new_classified_files):
                  ff.write(n+'<>'+new_files_classifiers[ij]+"\n")
            ff.close()
            classified_files_list = new_classified_files
	    files_classifier_list = new_files_classifiers
            del new_classified_files
            del new_files_classifiers
            return

        if event.key == 'escape':
             #end of classification
             #convex hull points are coming from classifier_dict[classifier_is]
             if os.path.isfile('classified_files.txt'):
	         f = open('classified_files.txt', 'a')
             else:
	         f = open('classified_files.txt', 'w+')
             convex_hull_points_as_array = np.asarray(convex_hull_points)
	     #print convex_hull_points_as_array
             hull = ConvexHull(convex_hull_points_as_array)
             #print ("Hull vertices = ",hull.vertices)
             hull_path = Path(convex_hull_points_as_array[hull.vertices])
            # print('hullpath =', hull_path)
             for i,p in enumerate(coordinates):
		 #print(i)
  #               print ("p[0] = ",p[0]," and p[1] = ",p[1])
#                 if p[0]>=-30 and p[0]<=-40:
 #                      if p[1]>=0 and p[1]<=2:
              #            print ("Center point of my w = ",p)
		 #print(p)
                 if hull_path.contains_point(p):
		      f.write(contourfiles[i]+'<>'+classifier_is+"\n")
		      classified_files_list.append(contourfiles[i])
		      files_classifier_list.append(classifier_is)
	     f.close()
	     del convex_hull_points
             convex_hull_points = [] 
        else:
            classifier_is = event.key
            
       # print ('k = ',classifier_is)


fig, ax = plt.subplots()
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkeypress)

def scatter_graph(contours_path,ax,fig):
	"""
        This function makes the scatter graph using bhsne. The barnes hut stochastic neighborhood embedding algorithm is superior to regular tsne because bhsne scales. The barnes hutt performs the n-body simulation using oct trees in the three dimensional space. This makes the algorithm much faster and able to scale to a larger dataset. For large datasets bhsne is better.

        The link https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding describes tsne in detail. The parameter for the bh_sne are:
        perplexity
        theta

        The perplexity is 2 to the entropy of the probability distribution. It measures how many neighbors each data point will be connected to. When I raise the perplexity the images have more clusters.
        Theta measures the accuracy of the algorithm. It is the angle the data points are to each other. Large theta speeds up the algorithm but reduces the accuracy and small theta slows down the algorithm but increases the accuracy.
        """
	ctr = 0
	ctr1=0
	contourlist = []
	global contourfiles
	global classified_files_list
        #read all the classfied files into a list
        #not only read the classified files into a list, but also keep it open for appending
	for dirName, subdirList, fileList in os.walk(contours_path):
		   #then filelist will be all images 
                   #read the files that have been classified
                   #path to contour and the classification
		   if ctr >10000:
		       break

		   if 'contours' in os.path.basename(dirName):
 		       for contourfile in fileList:
#			       if ctr1 % 5000 == 0:
#			           convex_hull_points = open('convex_hull_points_' +str(ctr1+1) +'.txt', 'r')
#			       ctr1+=1		        
              	               #fn = dirName+"/"+contourfile
                               #if contours_path in classified_files:
                               #we want to make sure contours_path is the full path of the contours file, if contours_path is *NOT* a full path, make sure to have a variable that is the full path
                               #contours_path_full_file_name = <whatever needs to be appeneded to the contours_path as ncessary>
                               #    continue

              	               fn = dirName+"/"+contourfile
                               if fn in classified_files_list:
#			           print( 'skipping', fn)
                                   continue
                
			       contourfiles.append(dirName+"/"+contourfile)
			       contour = np.fromfile(dirName+"/"+contourfile, dtype =np.uint8)
			       if contour.size != 784:
			           continue
			       contour = np.true_divide(contour,255)
			       contourlist.append(contour)
			       ctr+=1

	contourarray = np.asarray(contourlist)
        #print (contourarray)
	global coordinates
	coordinates = bh_sne(contourarray, perplexity = 30, theta = .1) * 10
	xs = []
	ys = []
	for coordinate in coordinates:
		xs.append(coordinate[0])
		ys.append(coordinate[1])
#	read_points_and_draw_convex_hull.read_points_and_draw_convex_hull()
	images_plot.plot_embedding(contourfiles, xs, ys, .63,ax)
#	read_points_and_draw_convex_hull.read_points_and_draw_convex_hull()
	plt.axis([-500,500,-500,500])
	plt.show()














if __name__ == '__main__':	
    parser = argparse.ArgumentParser(description = "path name")
    parser.add_argument("--logging_level",type=int)
    parser.set_defaults(logging_level = logging.INFO)
    #arg for figure_size, zoom
    parser.add_argument("--figure_size", type=tuple)
    parser.set_defaults(figure_size=(30,10))
    parser.add_argument("--zoom", type=int)
    parser.set_defaults(zoom=.9)
    parser.add_argument("--contours_path")
    parser.set_defaults(contours_path="/home/ubuntu/solver/images_output/") 
    args = parser.parse_args()
    #log = logging.getLogger(__name__)
    
    #points = [(1,1), (2,2), (3,3), (4,4), (5,5)]
    x = np.random.random(5)
    y = np.random.random(5)
    contourspath = '/home/ubuntu/solver/images_output/img2444/contours/'
    images=[contourspath+'4.png', contourspath+'10.png',contourspath+'11.png',contourspath+'12.png',contourspath+'13.png']
    images=[contourspath+'2.png', contourspath+'3.png',contourspath+'4.png',contourspath+'5.png',contourspath+'6.png']
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
    #plt.figure(figsize=args.figure_size)
    #lot_embedding(images, x,y, args.zoom)
    #if os.path.exists('reverse_classified_file_name'):
            #open file, read it entirely into a dictionary
            #each record is of the format <filename>,classifier
            #using a dictionary classified_files[filename] = classifier
    #else:
           #create a file 
           #and dictionary will not have anything in it
    if os.path.isfile('classified_files.txt'):
          temporary_classified_files_handle = open('classified_files.txt','r')
    else:
          temporary_classified_files_handle = open('classified_files.txt','w+')
          
    temporary_classified_files_list = temporary_classified_files_handle.readlines()
    temporary_classified_files_handle.close()
    classified_files_list = []
    files_classifier_list = []
    for f in temporary_classified_files_list:
         f=f.rstrip('\n')
	 classified_files_list.append(f.split('<>')[0])
	 files_classifier_list.append(f.split('<>')[1])
    scatter_graph(args.contours_path,ax,fig)
    #plt.show()
