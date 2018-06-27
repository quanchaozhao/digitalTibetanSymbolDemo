# coding:utf-8


import os

import numpy as np
from skimage import transform, io
from Tibetan_index.Demo_Test_Character_64 import get_image

def get_string_by_img(imgs, words_text='Tibetan_index\\Tibetan_symbol.txt', model=None):
    data = []
    for img in imgs:
        if len(img) == 0:
            continue
        # plt.subplot(211).imshow(img,cmap='gray')
        # plt.subplot(212).imshow(get_image(img),cmap='gray')
        # io.imsave('1.png',img.astype(np.uint8)*255)
        # io.imsave('2.png',(get_image(img)).astype(np.uint8))
        # plt.show
        # img = np.where(get_image(img) > 240, 255,0).astype(np.uint8)
        # img = transform.resize(img / 255,(64,32))
        img = transform.resize(img, (64, 32))
        data.append(img)
    data = np.asarray(data)
    data = np.reshape(data, (-1, 64, 32, 1))
    if model is None:
        return
    text_unicode = model.predict(data)

    word_dict = []
    with open(words_text, 'r') as f:
        for line in f.readlines():
            word_dict.append(line.strip())

    out_char_list = []
    for i, s_char in enumerate(text_unicode):
        char_uni = word_dict[np.argmax(s_char)]
        tem = []
        tem.append(char_uni)
        try:
            if 47 < int(char_uni, 16) < 58:
                arr_temp = np.argsort(s_char)
                char_uni = word_dict[arr_temp[1]]
                # out_char_list.append(chr(int(char_uni, 16)))
                out_char_list.append(tem)

            else:
                # out_char_list.append(chr(int(char_uni, 16)))
                out_char_list.append(tem)
        except Exception as e:

            print(i, e)

    return (out_char_list)


if __name__ == '__main__':
    path = r'D:\Users\Riolu\Desktop\t33'
    files = os.listdir(path)
    imgs = []
    for file in files:
        print(file)
        img = io.imread(os.path.join(path, file))
        imgs.append(img)
    print(get_string_by_img(imgs, words_text='..\\tools\\words_Titan.txt', model_path='..\\tools\\model_lenet5_v2.h5'))
