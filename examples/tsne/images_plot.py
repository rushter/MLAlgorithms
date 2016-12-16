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
#mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import png

#def onclick(event):
#        #f.write('xdata=%f, ydata=%f\n' %
#        #  (event.xdata, event.ydata))
#        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#          (event.button, event.x, event.y, event.xdata, event.ydata))
#def onkeypress(event):
#        k = event.key
#        print ('key=',k)





log = logging.getLogger()
#fig,ax=plt.subplots()

def array_to_png(nparray):
        return Image.fromarray(nparray,'L')

def plot_embedding(images, x,y,  zoom,ax):
       # fig.canvas.mpl_connect('button_press_event', onclick)
       # fig.canvas.mpl_connect('key_press_event', onkeypress)
	label_points = []
	source_files = []
	points = open('points.txt', 'w+')
	for i,xx in enumerate(x):
		np_img= np.fromfile(images[i], dtype=np.uint8)
                np_img = np_img.reshape(28,28) #.astype(dtype=np.float64)
		#ab = imscatter(x[i],y[i], array_to_png(np.true_divide(np_img,255)),figure_size, zoom)
		ab = imscatter(x[i],y[i], np_img, zoom)
                ax.add_artist(ab)
		label_points.append((x[i],y[i]))
		source_files.append(images[i])
	for i, contour in enumerate(source_files):
		points.write("%s " % str(label_points[i]))
		points.write("%s\n" % images[i])
	points.close()



def imscatter(x,y, img,zoom):
	#ax.scatter(x,y)
        #dictionary you can pass which will describe the color etc.
	im = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, (x,y), xycoords='data',frameon=False, boxcoords="data")  
        return ab

 

def coords(s):
        print ("s =",s," and type = ",type(s))
	x, y= map(int, s.split(","))
	return tuple(x,y)



if __name__ == '__main__':	
    parser = argparse.ArgumentParser(description = "path name")
    parser.add_argument("--logging_level",type=int)
    parser.set_defaults(logging_level = logging.INFO)
    #arg for figure_size, zoom
    parser.add_argument("--figure_size", type=tuple)
    parser.set_defaults(figure_size=(30,10))
    parser.add_argument("--zoom", type=int)
    parser.set_defaults(zoom=.7) 
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
    plot_embedding(images, x,y, args.zoom,ax)
