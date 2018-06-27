import os
import shutil
#from shutil import *

def select_sample_100(dirpath):

    for parent, dirnames, filenames in os.walk(dirpath):
        for dirname in dirnames:
            
            subdir = os.path.join(parent, dirname) 
            count = 0
            os.makedirs( parent + '_val' + '/' + dirname )

            for subparent, subdirnames, subfilenames in os.walk(subdir):
                for subfilename in subfilenames:
                    
                    imgdir = parent + '/' + dirname + '/' +  subfilename
                    dst = parent +'_val' + '/' +  dirname + '/' 
                    if count < 30:
                        shutil.move(imgdir, dst)  
                        count = count + 1


if __name__ == "__main__":

    select_sample_100("merged_ultimate_new")                   
