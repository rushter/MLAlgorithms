import argparse
import logging
import os
import sys

def label_changer(file_name, from_chr, to_chr, change_indices):
        clf = open('classified_files.txt','r')
        clf_list = clf.readlines()
        clf_fn = []
        clf_class = []
        for i,j in enumerate(clf_list):
           (a,b) = clf_list[i].split('<>')
          
           clf_fn.append(a)
           clf_class.append(b)

        clf.close()

        

        allchrs = open(file_name,'r')
        allchrs_list = allchrs.readlines()

	filename_list = []
	classifier_list = []
        for i,j in enumerate(allchrs_list):
           (a,b) = allchrs_list[i].split('<>')
           filename_list.append(a)
           classifier_list.append(b)

        allchrs.close()
 
        
	change_indices_array = change_indices.split(',')
	for i, j in enumerate(change_indices_array):
                if filename_list[i] in clf_fn:
                       x = clf_fn.index(filename_list[int(j)-1])
                       clf_class[x] = to_chr+"\n"
                      
	f = open('classified_files.txt', 'w')
	for i,j in enumerate(clf_fn):
		f.write(clf_fn[i] + '<>' + clf_class[i])
	f.close()	
			








if __name__ == '__main__':	
    parser = argparse.ArgumentParser(description = "path name")
    parser.add_argument("--file_name")
    parser.add_argument("--from_chr")
    parser.add_argument("--to_chr")
    parser.add_argument("--change_indices")
    parser.add_argument("--logging_level", type=int)
    parser.set_defaults(logging_level = logging.INFO)
    args = parser.parse_args()
    file_name = args.file_name
    from_chr = args.from_chr
    to_chr = args.to_chr
    change_indices = args.change_indices
    log = logging.getLogger()
    
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
    label_changer(file_name, from_chr, to_chr, change_indices)   
