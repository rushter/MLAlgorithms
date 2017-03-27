import numpy as np
from PIL import Image,ImageOps,ImageDraw
import PIL
import sys
import random
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import os
import argparse
import logging
import math
import sys

ndx_string=""
log = logging.getLogger()

def onclick(event):
        global classifier_is
        global no_cols
        global ndx_string
        try:
            if classifier_is is None:
                #print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                #  (event.button, event.x, event.y, event.xdata, event.ydata))
                x = math.ceil(event.xdata/28)
                y = math.ceil(event.ydata/28)
                ndx = (x-1)*no_cols+y
                ndx_string = ndx_string + str(int(ndx))+","
                #print ("Ndx = ",ndx," x = ",x,y,no_cols)
                print ndx_string
                sys.stdout.flush()
        except (ValueError,RuntimeError, TypeError, NameError):
            pass

def onkeypress(event):
        global classifier_is
        classifier_is = event.key

def onmove(event):
        #print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #  (event.button, event.x, event.y, event.xdata, event.ydata))
        global classifier_is

        try:
            if classifier_is is None:
                z = ax.transData.transform_point([event.xdata, event.ydata])
                print ("Z = ",z,event.xdata,event.ydata)
        except (ValueError,RuntimeError, TypeError, NameError):
            pass


def is_prime(n):
  if n == 2 or n == 3: return True
  if n < 2 or n%2 == 0: return False
  if n < 9: return True
  if n%3 == 0: return False
  r = int(n**0.5)
  f = 5
  while f <= r:
    if n%f == 0: return False
    if n%(f+2) == 0: return False
    f +=6
  return True    

def composite(classifier,number):
        global no_cols
	classified_files = open('classified_files.txt')
	list_of_files = []
	classifiers = []
	for line in classified_files:
		line = line.rstrip('\n')
		x = line.split('<>')
		classifiers.append(x[1])
                if x[1]==classifier:
		    list_of_files.append('/home/ubuntu/solver/'+x[0])
	images_list = []
	cntr1 = 0
	cntr2 = 0
	cntr3 = len(list_of_files)
        if is_prime(cntr3):
           cntr3-=1
	cntr3-=number
        print(cntr3)
	for f in list_of_files:
		if os.stat(f).st_size != 784:
			continue
		imgis = np.fromfile(f,dtype=np.uint8).reshape(28,28)
 		cntr1+=1
		images_list.append(imgis)
	if is_prime(cntr1):
		cntr1 = cntr1-1
	cntr1-=number
	difference = 9999999999999
	rows = 1
        cols = 99999999999
        for i in range(cntr3):
	    if i >0:
	    	if cntr3 % i == 0:
			if cols - rows < difference and cols - rows > 0:
				rows = i
				difference = cols - rows
                        	cols = cntr1/rows
                       
            # is cntr3 divisible by i, if yes , then your matrix is divisor x quatient
	#cols = cntr1/rows
        cols1 = cols
        cols = rows
        no_cols = cols1
        rows = cols1
	print ("cols =",cols)
	for j in range(cntr1):
		if j%cols == 0:
			cntr2+=1
	print ("Rows = ",cntr2)
	alist = []
	list_of_lists = []
	for i in range(cntr1):
		if i%cntr2 == 0 and i> 0:
			list_of_lists.append(alist)
			alist=[]
		alist.append(images_list[i])
	list_of_lists.append(alist)
	vslist = []
	for l in list_of_lists:
		vslist.append(np.vstack(l)) # columns
        print ("There are ",len(vslist)," items in vslist")
	hslist = np.hstack(vslist) 
        print ("Type of hslist = ",type(hslist))
	imh1 = Image.fromarray(hslist)
	plt.imshow(imh1, cmap='gray')
	plt.show()
		











fig, ax = plt.subplots()
#fig.canvas.mpl_connect('motion_notify_event', onmove)
fig.canvas.mpl_connect('key_press_event', onkeypress)
fig.canvas.mpl_connect('button_press_event', onclick)

if __name__ == '__main__':	
    parser = argparse.ArgumentParser(description = "path name")
    parser.add_argument("--my_path")
    parser.add_argument("--root_path")
    parser.add_argument("--classifier")
    parser.add_argument("--number", type=int)
    parser.add_argument("--logging_level",type=int)
    parser.add_argument("--extract_and_print_contours_or_show_contours")
    parser.set_defaults(root_path = "/home/ubuntu/solver")
    parser.set_defaults(my_path = "/home/ubuntu/solver//")
    parser.set_defaults(number=0)
    parser.set_defaults(logging_level = logging.INFO)
    args = parser.parse_args()
    mypath = args.my_path
    root_path = args.root_path
    classifier = args.classifier
    number = args.number
    #log = logging.getLogger(__name__)
    
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
    composite(classifier, number)
