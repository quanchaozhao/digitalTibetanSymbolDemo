 
'''
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py
Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers.advanced_activations import PReLU
import random,cPickle
import theano
from sklearn.ensemble import RandomForestClassifier
import loadData_64_64 
import copy  
import sys   
sys.setrecursionlimit(1000000)


def rf(traindata,trainlabel,testdata,testlabel):
    print("Start training Random Forest...")
    rfClf = RandomForestClassifier(n_estimators=400,criterion='gini')
    rfClf.fit(traindata,trainlabel)
    
    pred_testlabel = rfClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-rf Accuracy:",accuracy)

def select_top5_class(pro_array):
	class_index = np.zeros((5, ), dtype = "int32")
	pro = np.zeros((5, ), dtype = "float32") 
	rows = pro_array.shape[0]	

	for j in range(rows):
		if pro_array[j] > pro[0]:
			pro[4] = pro[3]
			class_index[4] = class_index[3]
			pro[3] = pro[2]
			class_index[3] = class_index[2]
			pro[2] = pro[1]
			class_index[2] = class_index[1]
			pro[1] = pro[0]
			class_index[1] = class_index[0]
			pro[0] = pro_array[j]
			class_index[0] = j

		elif pro_array[j] > pro[1]:
			pro[4] = pro[3]
			class_index[4] = class_index[3]
			pro[3] = pro[2]
			class_index[3] = class_index[2]
			pro[2] = pro[1]
			class_index[2] = class_index[1]
			pro[1] = pro_array[j]
			class_index[1] = j

		elif pro_array[j] > pro[2]:
			pro[4] = pro[3]
			class_index[4] = class_index[3]
			pro[3] = pro[2]
			class_index[3] = class_index[2]
			pro[2] = pro_array[j]
			class_index[2] = j

		elif pro_array[j] > pro[3]:
			pro[4] = pro[3]
			class_index[4] = class_index[3]
			pro[3] = pro_array[j]
			class_index[3] = j

		elif pro_array[j] > pro[4]:
			pro[4] = pro_array[j]
			class_index[4] = j

	del pro
		
	return class_index	


def calu_top1_top2_top3_top5_acc(pro_arrays, labels):

	rows = pro_arrays.shape[0]
	count5 = float (0)
	count3 = float (0)
	count2 = float (0)
	count1 = float (0)
	for it in range(rows):
		top5_class = select_top5_class(pro_arrays[it])
		#print ("top_5:", top5_class)
		#print("label:", labels[it])
		for ite in range(5):
			if top5_class[ite] == labels[it]:
				count5 = count5 + 1
				if ite == 0:
					count1 = count1 + 1
				if ite < 2:
					count2 = count2 + 1
				if ite < 3:
					count3 = count3 + 1
				break
	
	return (float)(count1 / rows), (float)(count2 / rows), (float)(count3 / rows), (float)(count5 / rows)	
	
	
batch_size = 128
nb_classes = 946
nb_epoch = 150

# input image dimensions
img_rows, img_cols = 64, 64
# number of convolutional filters to use
#nb_filters = 32
# size of pooling area for max pooling
#nb_pool = 2
# convolution kernel size
#nb_conv = 3

# the data, shuffled and split between tran and test sets
X_train, y_train, X_test, y_test = loadData_64_64.read_image_from_files( )

print ("y_test:" ,y_test)
print ("y_test_type:", y_test.dtype)
#X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
train_size = X_train.shape[0]
val_size = X_test.shape[0]

'''
train_feature = np.empty((train_size, 250),dtype="float32")
val_feature = np.empty((val_size, 250),dtype="float32")
'''
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(100, 3, 3,
						init = 'uniform',
                        border_mode='valid', 
		  	            subsample=(1, 1),
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
#model.add(PReLU(init='zero', weights=None))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))


model.add(Convolution2D(200, 2, 2, border_mode='valid', init = 'uniform'))
model.add(Activation('relu'))
#model.add(PReLU(init='zero', weights=None))

#model.add(Dropout(0.10))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
#model.add(Dropout(0.25))
#model.add(ZeroPadding2D(padding=(1,1), dim_ordering='th'))


model.add(Convolution2D(300, 2, 2, border_mode='valid', init = 'uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))


model.add(Convolution2D(400, 2, 2, border_mode='valid', init = 'uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
'''
model.add(Convolution2D(750, 2, 2, border_mode='valid', init = 'uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
'''


#model.add(PReLU(init='zero', weights=None))i
'''
model.add(ZeroPadding2D(padding=(2,2), dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

model.add(Convolution2D(360, 3, 3, init = 'uniform'))
#model.add(Activation('relu'))
model.add(PReLU(init='zero', weights=None))
model.add(Dropout(0.20))

#model.add(ZeroPadding2D(2,2), dim_ordering='th')
#model.add(MaxPooling2D(pool_size=(3, 3), stride(2,2)))
'''

model.add(Flatten())

model.add(Dense(500, init = 'uniform'))
model.add(Activation('relu'))
#model.add(MaxoutDense(500, nb_feature=4, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, input_dim=None))
#model.add(PReLU(init='zero', weights=None))
#model.add(Dropout(0.5))

#model.add(Dense(250))
#model.add(Activation('relu'))
#model.add(MaxoutDense(250, nb_feature=4, init='uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, input_dim=None))
#model.add(PReLU(init='zero', weights=None))
#model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.008, decay=1e-6,  momentum=0.9, nesterov=True), metrics=["accuracy"] )
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=60,show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
