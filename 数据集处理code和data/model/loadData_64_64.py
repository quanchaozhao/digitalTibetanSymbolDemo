#coding:utf-8
"""
Author: lji
fun: load image data to prepare for CNN recognition
file: loadData.py
use 4 spaces to fomat the source 
"""
import os
from PIL import Image
import numpy as np
import string
import sys
import random

def read_label_from_txt(path_txt, mod):
    image_names = []
    labels = []
    image_names_labels = []
    #strokenums = []
    #pointnums = []

    file = open (path_txt)
    for line in file:
        # print(line.rstrip().split(' ',2)
        #image_name_label = line.rstrip().split(' ',2)
        image_names_labels.append(line)
    file.close()


    if mod == 1 :	
        random.shuffle(image_names_labels)    
       
    for elem in image_names_labels:
        #print(elem
       	image_name_label = elem.rstrip().split(' ', 2)
        image_names.append(image_name_label[0])
        labels.append(image_name_label[1])
        #strokenums.append((float)(image_name_label[2]))
        #pointnums.append((float)(image_name_label[3]))		
	
    del image_names_labels
   
    return image_names, labels 


def read_image_from_files( ):
    
    print("load data from thenao_keras")

    dir_images_train = "../thenao_keras/train/"
    dir_images_val = "../thenao_keras/val/"
    dir_label = "../thenao_keras/label/"
    file_train_label = "train.txt"
    file_val_label = "val.txt"


    print("load" + (dir_label + file_train_label))
    train_images, train_labels  = read_label_from_txt( dir_label +file_train_label, 0)
    print("finish.....")

    print("load" + (dir_images_val + file_val_label))
    val_images, val_labels  = read_label_from_txt( dir_label +file_val_label, 0)
    print("finish....")

    print("train_size:" + str(len(train_images)))
    print("val_size:" + str(len(val_images)))
    
    train_size = len(train_images)
    val_size = len(val_images)
    train_data = np.empty((train_size, 1, 64, 64), dtype = "float32")
    #train_strokenums_pointnums = np.empty((train_size, 2), dtype = "float32")
    train_labels_ = np.empty((train_size), dtype = "int32")    
    val_data = np.empty((val_size, 1, 64, 64), dtype = "float32")
    #val_strokenums_pointnums = np.empty((val_size, 2), dtype = "float32")
    val_labels_ = np.empty((val_size), dtype = "int32") 
  
   # print(dir_images_train + train_images[1]
   
    print("load train data start.....")    
    for i in range(train_size):
        
        if (i + 1) % 500 == 0 or (i + 1)  == train_size:
            sys.stdout.write('\rdone:' + str(i + 1) + '/'+ str(train_size) )
            sys.stdout.flush()
        
        #if i <= 20000:
        img = Image.open(dir_images_train + train_images[i])
       	arr = np.asarray(img, dtype = "float32")
        #print(arr.shape
        #arr = arr.reshape(3, 128, 64)

        train_data[i, :, :, : ] = arr
        #train_strokenums_pointnums[i,:] = [train_strokenums[i], train_pointnums[i]]
        train_labels_[i] = int(train_labels[i])
    print("\nload train data finish.....")

    print("load val data start......")
    for i in range(val_size):
        
        if (i + 1) % 500 == 0 or (i + 1) == val_size:
            sys.stdout.write('\rdone:' + str(i + 1) + '/'+ str(val_size) )
            sys.stdout.flush()

        img = Image.open(dir_images_val + val_images[i])
        arr = np.asarray(img, dtype = "float32")
        #arr = arr.reshape(3, 128, 64)

        val_data[i, :, :, :] = arr
        #val_strokenums_pointnums[i,:] = [val_strokenums[i], val_pointnums[i]]
        val_labels_[i] = int(val_labels[i])
     
    print("\nload val data finish....")

    return train_data, train_labels_, val_data, val_labels_  



if __name__ == '__main__':
    read_image_from_files( )
