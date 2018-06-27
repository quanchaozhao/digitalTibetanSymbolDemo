import os
import os.path
import numpy as np
import cv2
from PIL import Image

global resizecount 

def mkdirFiles( dirPath, filePath):
    files = open(filePath)
    for line in files:
        filename = line.rstrip()
        os.mkdir(dirPath + "/" + filename)      
    files.close()

def  resize_rate(dirPath,  ratio, resizecount):
    #0.5 < ratio < 1
    #resizecounti
    for   parent,dirnames,filenames  in os.walk(dirPath):
        for filename in filenames:
            imgdir = parent + "/" + filename
            img = cv2.imread(imgdir)
            emptyImage = np.zeros(img.shape, np.uint8) 
            #emptyImage = emptyImage + 255
            img=cv2.resize(img, (int (img.shape[1] * ratio), int (img.shape[0] * ratio)), interpolation=cv2.INTER_CUBIC)
            height =  emptyImage.shape[0]
            width =  emptyImage.shape[1]
            for i in range( (int)(height - height *ratio) / 2, (int)(height + height *ratio) / 2):
                for j in range( (int)(width - width * ratio)  / 2, (int)(width + width * ratio)  / 2):
                    emptyImage[i,j] =  img[i -  (height - height *ratio) / 2, j - (width - width * ratio)  / 2]  
                               
            cv2.imwrite( parent + "/" + filename.split('.',2)[0] + str(resizecount) + ".png", emptyImage)

def   rotate_angle(dirPath, angle, ratio):

    for   parent,dirnames,filenames  in os.walk(dirPath):
        for filename in filenames:
            imgdir = parent + "/" + filename
            #print(imgdir)
            img = cv2.imread(imgdir)
            emptyImage = np.zeros(img.shape, np.uint8) 
            #emptyImage = emptyImage + 255  
            img=cv2.resize(img, (int (img.shape[1] * ratio), int (img.shape[0] * ratio)), interpolation=cv2.INTER_CUBIC)
            height =  emptyImage.shape[0]
            width =  emptyImage.shape[1]
            for i in range( (int)(height - height *ratio) / 2, (int)(height + height *ratio) / 2):
                for j in range( (int)(width - width * ratio)  / 2, (int)(width + width * ratio)  / 2):
                    emptyImage[i,j] =  img[i -  (height - height *ratio) / 2, j - (width - width * ratio)  / 2]  
            resizecount = 0                   
            cv2.imwrite( parent + "/" + filename.split('.',2)[0] + str(resizecount) + ".png",  emptyImage)
            img = Image.open( parent + "/" + filename.split('.',2)[0] + str(resizecount) + ".png")
            img = img.rotate(angle)
            img.save(parent + "/" + "r" + str(angle) +str(int(100 *ratio)) + '_' + filename )
            #os.remove(parent + "/" + filename.split('.',2)[0] + str(resizecount) + ".png" )  

# 增加验证集合数据
def   rotate_angle_val(dirPath, angle):

    for   parent,dirnames,filenames  in os.walk(dirPath):
        for filename in filenames:
            imgdir = parent + "/" + filename
            #img = cv2.imread(imgdir)
            '''
            emptyImage = np.zeros(img.shape, np.uint8) 
            img=cv2.resize(img, (int (img.shape[1] * ratio), int (img.shape[0] * ratio)), interpolation=cv2.INTER_CUBIC)
            height =  emptyImage.shape[0]
            width =  emptyImage.shape[1]
            for i in range( (int)(height - height *ratio) / 2, (int)(height + height *ratio) / 2):
                for j in range( (int)(width - width * ratio)  / 2, (int)(width + width * ratio)  / 2):
                    emptyImage[i,j] =  img[i -  (height - height *ratio) / 2, j - (width - width * ratio)  / 2]  
                               
            cv2.imwrite( "train_data" + "/" + filename.split('.',2)[0] + "/" +  filename, emptyImage)
            '''
            img = Image.open(imgdir)
            img = img.rotate(angle)
            img.save("val_data_s" + "/" + filename.split('.',2)[0] + "/" + "r" + str(angle) + '_' + filename )  
  
def   delete_files(dirPath):

    for parent, dirnames, filenames  in os.walk(dirPath):
        for dirname in dirnames:
            imgdir = os.path.join( parent, dirname, dirname +'.jpg')
            os.remove(imgdir)
             
            


def  saltPeperNoise( img, ratio):

    noisenum = (int)(img.shape[0]*img.shape[1]*ratio)
    height = img.shape[0]
    width = img.shape[1]

    for i in range(noisenum):
        randx =  np.random.random_integers(0, height - 1)
        randy =  np.random.random_integers(0, width - 1)
        
        if np.random.random_integers(0, 1) == 0:
            img[randx, randy] = 0
        else:
            img[randx, randy] = 255  
            
    return img

def  GaussNoise(img,  sigma):

    mean = 0
    height = img.shape[0]
    width = img.shape[1] 

    for row in range(height):
        for col in range(width):

            value = float(img[row, col])
            value += np.random.normal(mean, sigma)
            #print(value)
      
            if value < 0:
                value = 0
            if value > 255:
                value = 255        
            img[row, col] = (int)(value)
            #print(img[row, col] )
    return img
   
def  addNoiseToImage(dirPath, angle, ratio, mod):
    
    for parent,dirnames,filenames  in os.walk(dirPath):
        for filename in filenames:
            imgdir = parent + "/" + filename
            img = cv2.imread(imgdir)
            '''
            emptyImage = np.zeros(img.shape, np.uint8) 
            #emptyImage
            img=cv2.resize(img, (int (img.shape[1] * ratio), int (img.shape[0] * ratio)), interpolation=cv2.INTER_CUBIC)
            height =  emptyImage.shape[0]
            width =  emptyImage.shape[1]
            for i in range( (int)(height - height *ratio) / 2, (int)(height + height *ratio) / 2):
                for j in range( (int)(width - width * ratio)  / 2, (int)(width + width * ratio)  / 2):
                    emptyImage[i,j] =  img[i -  (height - height *ratio) / 2, j - (width - width * ratio)  / 2]  
                               
            cv2.imwrite( "train_data_s" + "/" + filename.split('.',2)[0] + "/" +  filename, emptyImage)
            img = Image.open("train_data_s" + "/" + filename.split('.',2)[0] + "/" +  filename)
            img = img.rotate(angle)
            img.save("train_data_s" + "/" + filename.split('.',2)[0] + "/" +  filename)
            img = cv2.imread("train_data_s" + "/" + filename.split('.',2)[0] + "/" +  filename)
            '''
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if mod == 0: 
                img = saltPeperNoise( img, 0.1)         
            else:
                img =  GaussNoise(img, 60)
            cv2.imwrite( parent + "/" + "r_" + str(mod) +"n_" + str(angle) +str(100 *ratio ) +"_" + filename, img)


def  resize_image(dirPath):

    for   parent, dirnames, filenames  in os.walk(dirPath):
        for dirname in dirnames:

            subdir = os.path.join(parent, dirname)

            for subparent, subdirnames, subfilenames in os.walk(subdir):
                for subfilename in subfilenames: 
    
                    imgdir = subdir + "/" + subfilename
                    img = cv2.imread(imgdir)
                    #img=cv2.resize(img, (32, 64), interpolation=cv2.INTER_CUBIC)           
                    retval, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV) 
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(imgdir, img)

def  add_image(dirPath):

    for   parent, dirnames, filenames  in os.walk(dirPath):
        for dirname in dirnames:

            subdir = os.path.join(parent, dirname)
            num = len(os.listdir( subdir ) )
            print(num) 
                  
            if num >= 150: 
                continue

            else:

                if num  > 50:
                    resizecount  = 2
                    #resize_rate( subdir, 0.9, resizecount)
                    rotate_angle(subdir, -5, 0.8)
                   # addNoiseToImage(subdir, 0,0,0)  
                elif num > 25:

                    resizecount  = 2
                    resize_rate( subdir, 0.9, resizecount)
                    rotate_angle(subdir, -20, 0.8)

                else:
                    resizecount  = 2
                    resize_rate( subdir, 0.9, resizecount)
                    rotate_angle(subdir, -20, 0.8)
                    addNoiseToImage(subdir, 0,0,0) 
                   
                         
                     


if __name__ == "__main__":
  
     #mkdirFiles( "train_data_s",  "words_Titan.txt")
     #3mkdirFiles( "val_data_s",  "words_Titan.txt") 
     '''
     resizecount  = 2
     resize_rate( "tibetanfont0", 0.9, resizecount)
     resizecount  = 3
     resize_rate( "tibetanfont0", 0.85, resizecount)
     resizecount  = 4
     resize_rate( "tibetanfont0", 0.8, resizecount)
     resizecount  = 5
     resize_rate( "tibetanfont0", 0.75, resizecount)
     resizecount  = 6
     resize_rate( "tibetanfont0", 0.7, resizecount)
     resizecount  = 7
     resize_rate( "tibetanfont0", 0.65, resizecount)
     resizecount  = 8
     resize_rate( "tibetanfont0", 0.6, resizecount)
     resizecount  = 9
     resize_rate( "tibetanfont0", 0.55, resizecount)
     '''
     #rotate_angle("tibetanfont0", -10, 0.8)
     '''
     for  i in range(10, 18):
          rotate_angle("tibetanfont0s", 25, i * 0.05)
          rotate_angle("tibetanfont0s", -25, i * 0.05)
          #rotate_angle("tibetanfont0s",  15, i * 0.05)
          #rotate_angle("tibetanfont0s", -15, i * 0.05)
     '''
     '''
     for  i in range(10, 18):
          addNoiseToImage("tibetanfont0s", 20, i * 0.05,  0) 
          addNoiseToImage("tibetanfont0s", 20, i * 0.05,  1) 
          addNoiseToImage("tibetanfont0s", -20, i * 0.05,  0)
          addNoiseToImage("tibetanfont0s", -20, i * 0.05,  1) 
     '''
     #addNoiseToImage("tibetanfont0", 10, 0.85, 1)
     #rotate_angle_val("tibetanfont0s", 20)
     #rotate_angle_val("tibetanfont0s", -20)    
     #rotate_angle_val("tibetanfont0s", 15)
     #rotate_angle_val("tibetanfont0s", -15)
     #delete_files("train_data_s")
     #val_words
     resize_image("merged_ultimate_new")
     #add_image("merged_ultimate_new")
      
