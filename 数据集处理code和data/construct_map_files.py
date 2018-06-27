import os
import numpy as np


def getWordDict(filepath):

    files = file(filepath)
    id_ = 0
    word2IdDict = {}        
    Id2WordDict = {}
    for line in files:

        word = line.strip('\n')
        word2IdDict[word] = id_
        Id2WordDict[id_] = word
        id_ = id_ + 1
    
    #print(word2IdDict)
    
    return word2IdDict, Id2WordDict


def writeLabelFile(dirpath, filepath):

    wordDict, id2wordDict = getWordDict("words_Titan.txt")
    wfiles =file(filepath,"w")     

    for parent, dirnames, filenames in os.walk(dirpath):
        for dirname in dirnames:

            subdir = os.path.join(parent, dirname)
            id_ = wordDict[dirname] 

            for subparent, subdirnames, subfilenames in os.walk(subdir):
                for subfilename in subfilenames:

                    imgdir = parent + '/' + dirname + '/' +  subfilename  + ' ' + str(id_) +'\n'
                    wfiles.write(imgdir)
    wfiles.close()                                    
        


if __name__ == "__main__":
 
    getWordDict("words_Titan.txt")   
    writeLabelFile("train_data", "train.txt")
    writeLabelFile("val_data", "val.txt")
