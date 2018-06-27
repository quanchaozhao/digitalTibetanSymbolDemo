# coding:utf-8

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
import keras
import os

input_img = Input(shape=(60, 60, 1))

x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8,  kernel_size=(3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

np_file = np.load("/home/lyx2/data.npz")
x_train = np_file['x_train']
y_train = np_file['y_train']
x_train = x_train.astype('float32')
x_train = np.reshape(x_train, (len(x_train), 60, 60 ,1))


from keras.callbacks import TensorBoard
if not os.path.exists("encoder.h5"):
    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,)
    encoder = Model(input_img,encoded)
    encoder.save("encoder.h5")
else:
    encoder = keras.models.load_model("encoder.h5")



