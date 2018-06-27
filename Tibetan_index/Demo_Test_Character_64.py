# coding:utf-8
import os

import keras.models as models
import numpy as np
from skimage import transform, io
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io,transform

MAXROW = 64
MAXCOL = 32
import cv2

def resize_image(path,i):
    filename = os.path.join(path, i)
    image = (io.imread(filename,as_grey=True)).astype(np.uint8)
    image = np.where(image>0,0,1)
    tem_count = np.sum(image,axis=0)
    index = np.where(tem_count>0)
    col_min, col_max = index[0][0],index[0][-1]
    tem_count = np.sum(image, axis=1)
    index2 = np.where(tem_count>0)
    row_min, row_max = index2[0][0], index2[0][-1]
    image_sub = image[row_min:(row_max + 1), col_min: (col_max + 1)]

    # print('index:',str(row_min),str(row_max + 1), str(col_min), str(col_max + 1))
    image_sub = np.where(image_sub>0,0,1)
    return (image_sub*255).astype(np.uint8)

def format_image_size(path,i):
    filename = os.path.join(path, i)
    image = (io.imread(filename,as_grey=True)).astype(np.uint8)
    image = np.where(image>0,0,1)

    tem_count = np.sum(image,axis=0)
    index = np.where(tem_count>0)
    col_min, col_max = index[0][0],index[0][-1]
    tem_count = np.sum(image, axis=1)
    index2 = np.where(tem_count>0)
    row_min, row_max = index2[0][0], index2[0][-1]
    image_sub = image[row_min:(row_max + 1), col_min: (col_max + 1)]
    image_sub = np.where(image_sub > 0, 0, 1)
    image = (image_sub * 255).astype(np.uint8)
    row, col = image.shape
    # image_TT = cv2.resize(image,(MAXCOL, MAXROW),interpolation=cv2.INTER_CUBIC).astype(np.uint8)
    # io.imsave(os.path.join(r'C:\Users\Zqc\Desktop\Just_resize\formate',i),image_TT)
    # return 0
    # print(i,[row, col])
    if(row < MAXROW  and col < MAXCOL):
        count = 1
        while(row * count < MAXROW and col * count < MAXCOL):
            count = count + 1
        if(count > 1):
            count = count - 1
        if count > 1:
            new_image = cv2.resize(image,(col * count, row * count),interpolation=cv2.INTER_CUBIC).astype(np.uint8)
        else:
            new_image = image
        new_bg = np.ones((MAXROW,MAXCOL)) * 255
        row_init = int((MAXROW - row*count) / 2)
        col_init = int((MAXCOL - col*count) / 2)
        # new_image = np.where(new_image > 240,0,255)
        new_bg[row_init:(row_init + row*count), col_init:(col_init + col*count)] = new_image
        # print("DDDDDDDDDDDDDDDDDDDD")
    else:
        count = 1
        while(row / count > MAXROW or col / count > MAXCOL):
            count = count + 1
        # count = count - 1
        new_image = image
        if count > 1:
            # print("DDDDDDDD")
            new_image = cv2.resize(image, (int(col / count), int(row / count)), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
        new_bg = np.ones((MAXROW, MAXCOL)) * 255
        row_init = int((MAXROW - row / count) / 2)
        col_init = int((MAXCOL - col / count) / 2)
        # new_image = np.where(new_image > 240, 0, 255)
        new_bg[row_init:int(row_init + row / count), col_init:int(col_init + col / count)] = new_image

    # plt.subplot(111).imshow(new_bg.astype(np.uint8), cmap='gray')
    # plt.show()

    img = (new_bg.astype(np.uint8))
    # io.imsave(os.path.join(r'C:\Users\Zqc\Desktop\Demo\format',i),img)
    return img

def get_string_by_img(path, i, img,words_text='./Tibetan_symbol.txt', model_path='./model_lenet5_v2_extend.h5'):
    # img = io.imread(os.path.join(path, i), as_grey=True)
    img = transform.resize(img, (64, 32))
    data = []
    data.append(img)
    data = np.asarray(data)
    data = np.reshape(data, (-1, 64, 32, 1))

    text_code = model.predict(data)

    word_dict = []
    with open(words_text, 'r') as f:
        for line in f.readlines():
            word_dict.append(line.strip())

    code = word_dict[np.argmax(text_code)]
    # if not(i[0:6] == code):
    print("原始图像类别：", i[0:6], "预测图像类别：", code)
    i = i[0:6]
    return i == code

    # data = []
    # for img in imgs:
    #     if len(img) == 0:
    #         continue
    #     img = transform.resize(img,(64,32))
    #     data.append(img)
    # data = np.asarray(data)
    # data = np.reshape(data,(-1,64,32,1))
    # import keras.models
    # model = keras.models.load_model(model_path)
    # text_unicode = model.predict(data)
    #
    # word_dict = []
    # with open(words_text,'r') as f:
    #     for line in f.readlines():
    #         word_dict.append(line.strip())
    #
    # out_char_list = []
    # for i,s_char in enumerate(text_unicode):
    #     char_uni = word_dict[np.argmax(s_char)]
    #     try:
    #         if 47 < int(char_uni,16) < 58:
    #             arr_temp = np.argsort(s_char)
    #             char_uni = word_dict[arr_temp[1]]
    #             out_char_list.append(chr(int(char_uni, 16)))
    #         else:
    #             out_char_list.append(chr(int(char_uni,16)))
    #     except Exception as e:
    #         print(i,e)
    #
    # return ''.join(out_char_list)

def get_image(img):
    image = np.where(img<1,1,0)
    tem_count = np.sum(image, axis=0)
    index = np.where(tem_count > 0)
    col_min, col_max = index[0][0], index[0][-1]
    tem_count = np.sum(image, axis=1)
    index2 = np.where(tem_count > 0)
    row_min, row_max = index2[0][0], index2[0][-1]
    image_sub = image[row_min:(row_max + 1), col_min: (col_max + 1)]
    image_sub = np.where(image_sub > 0, 0, 1)
    image = (image_sub * 255).astype(np.uint8)
    row, col = image.shape
    # image_TT = cv2.resize(image,(MAXCOL, MAXROW),interpolation=cv2.INTER_CUBIC).astype(np.uint8)
    # io.imsave(os.path.join(r'C:\Users\Zqc\Desktop\Just_resize\formate',i),image_TT)
    # return 0
    # print(i,[row, col])
    if (row < MAXROW and col < MAXCOL):
        count = 1
        while (row * count < MAXROW and col * count < MAXCOL):
            count = count + 1
        if (count > 1):
            count = count - 1
        if count > 1:
            new_image = cv2.resize(image, (col * count, row * count), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
        else:
            new_image = image
        new_bg = np.ones((MAXROW, MAXCOL)) * 255
        row_init = int((MAXROW - row * count) / 2)
        col_init = int((MAXCOL - col * count) / 2)
        # new_image = np.where(new_image > 240,0,255)
        new_bg[row_init:(row_init + row * count), col_init:(col_init + col * count)] = new_image
        # print("DDDDDDDDDDDDDDDDDDDD")
    else:
        count = 1
        while (row / count > MAXROW or col / count > MAXCOL):
            count = count + 1
        # count = count - 1
        new_image = image
        if count > 1:
            # print("DDDDDDDD")
            new_image = cv2.resize(image, (int(col / count), int(row / count)), interpolation=cv2.INTER_CUBIC).astype(
                np.uint8)
        new_bg = np.ones((MAXROW, MAXCOL)) * 255
        row_init = int((MAXROW - row / count) / 2)
        col_init = int((MAXCOL - col / count) / 2)
        # new_image = np.where(new_image > 240, 0, 255)
        new_bg[row_init:int(row_init + row / count), col_init:int(col_init + col / count)] = new_image

    # plt.subplot(211).imshow(img, cmap='gray')
    # plt.subplot(212).imshow(new_bg.astype(np.uint8), cmap='gray')
    # plt.show()

    img = (new_bg.astype(np.uint8))
    # io.imsave(os.path.join(r'C:\Users\Zqc\Desktop\Demo\format',i),img)
    return img
path = r'C:\Users\Zqc\Desktop\character\character'
# path = r'C:\Users\Zqc\Desktop\Demo\format'
# path = r'C:\Users\Zqc\Desktop\TT\re'
# path = r'C:\Users\Zqc\Desktop\character\format'
# path = r'C:\Users\Zqc\Desktop\character\resize'
# count = 0
# tem = 0
# model_path = './model_lenet5_v2_extend.h5'
# model = models.load_model(model_path)
# if __name__ == '__main__':
#     get_image()

#     for i in os.listdir(path):
#         img = format_image_size(path, i)
#         tem = tem + get_string_by_img(path, i, img) * 1
#         count = count + 1
#     print('Correct character = ', tem, ', total character = ', count, ',Accuracy: ', tem / count * 100, '%')
from keras import backend as K


K.clear_session()