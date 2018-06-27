# coding:utf-8
from __future__ import print_function,division
from pic_process.img_process import line_extract, char_extract

IN_PATH = "/home/lyx2/img_in"
CHAR_OUT_PATH = r"D:\Users\Riolu\Desktop\aabbcc"
CLASSIFY_OUT_PATH = R"D:\npq"
MERGED_PATH = r"D:\merged"
SORTED_PATH = r"D:\sorted"
WRAP_SIZE = 60
# 这里用的数据是经过倾斜校正，噪点去除，文本区域检测后的数据
# 生成文本并进行初步的归类 归类的方法是按照 余弦相似度


import os
from shutil import copyfile,rmtree,copytree
import numpy as np
from skimage import io
from skimage import transform
from skimage import util
from sklearn.metrics.pairwise import cosine_similarity


TH = 0.9

# 比较两个图片的余弦相似度
def cosine_similarity_2(img1,img2):

    img1 = np.asarray(img1)
    img2 = np.asarray(img2)

    return cosine_similarity(img1.reshape((1,-1)), img2.reshape((1,-1))).ravel()[0]

# 将文件夹下相似度大于 TH 的文件归并成一类，相同类的路径名称相同，用于对于切分好的文字进行分类。
# 可以替换成其他分类方法
def classify_image_by_cosine_similarity(path_in=CHAR_OUT_PATH,path_out=CLASSIFY_OUT_PATH):
    counter = 0
    sample_img_list = []
    sample_file_name_list = []
    total = len(os.listdir(path_in))
    for i,file_name in enumerate(os.listdir(path_in)):
        img_path = path_in + os.path.sep + file_name
        if not os.path.isdir(img_path):
            img1 = io.imread(img_path, as_grey=True)
            img1 = util.invert(img1)
            img1 = transform.resize(img1, (100, 100),mode="edge")
            for index, img in enumerate(sample_img_list):
                if cosine_similarity_2(img, img1) > TH:
                    sample_file_name_list[index].append(file_name)
                    break
            else:
                img1 = transform.resize(img1, (100, 100),mode="edge")
                sample_img_list.append(img1)
                sample_file_name_list.append([file_name])
                counter += 1
        if i % 10 == 0:
            print("%d of %d" % (i,total))

    for i, file_names in enumerate(sample_file_name_list):
        out_dir = path_out + os.path.sep + str(i)
        os.mkdir(out_dir)
        for file_name in file_names:
            src = path_in + os.path.sep + file_name
            dst = out_dir + os.path.sep + file_name
            copyfile(src, dst)
    return sample_img_list

# 将目录下的文件按照目录名称生成数据集  100个作为训练集，20个作为测试集
def extract_exp_data_by_path_name(path_in,wrap_size=100,to_npz=True):
    data = None
    for file_dir in os.listdir(path_in):
        img_dir_path = os.path.join(path_in,file_dir)
        if os.path.isdir(img_dir_path):
            files = os.listdir(img_dir_path)
            file_numbers = len(files)
            print(file_numbers)
            if file_numbers < 15:
                if data is None:
                    data = []
                    for file in files:
                        img_file_path = os.path.join(img_dir_path,file)
                        img = io.imread(img_file_path,as_grey=True)
                        img = util.invert(img)
                        data.append(img)
                print("need copy to 15")
                while file_numbers < 15:
                    pass

            if file_numbers < 30:
                print("need to blur")

            if file_numbers < 60:
                print("need to rotation")

            if file_numbers < 120:
                print("need to add noise")
    pass

# 将目录下的目录中的文件按照目录的名称标记数据并生成数据集
def label_image_by_path_name(path_in,index_list=None,shuffle=True,wrap=True,wrap_size=60,to_npz=True):
    '''
    将path_in目录下的图片文件按照目录，生成数据集
    :param path_in: 已经按照文件目录分好的图片
    :param shuffle: 是否打乱
    :param index_list 如果给定 index_list则会按照index_list中编码出现的顺序，将数据目标label标为数据
    :param to_npz 是否导出到npz文件
    :return: 
    例如： 路径名为 'a','b','c' index_list=['b','c','a'] 则'a'路径下的所有的样本将被标记为2，'b'为0，'c'为1
    '''
    data = []
    for file_dir in os.listdir(path_in):
        img_dir_path = path_in + os.path.sep + file_dir
        if os.path.isdir(img_dir_path):
            label = file_dir
            if index_list is not None:
                try:
                    label = index_list.index(file_dir)
                except Exception as e:
                    print(e)

            for i, file_name in enumerate(os.listdir(img_dir_path)):
                img_file_path = img_dir_path + os.path.sep + file_name
                if not os.path.isdir(img_file_path):
                    img = io.imread(img_file_path,as_grey=True)
                    img = util.invert(img)
                    if wrap:
                        img = transform.resize(img,(wrap_size,wrap_size),mode='reflect')
                    data.append([img,label])
                if i > 100:
                    break
    # 打乱数据
    if shuffle:
        np.random.shuffle(data)

    x_data = []
    y_data = []
    for img,label in data:
        x_data.append(img)
        y_data.append(label)

    if to_npz:
        x_train = np.asarray(x_data)
        y_train = np.asarray(y_data)
        print(x_train.shape)
        print(y_train.shape)
        np.savez("D:\\data.npz",x_train=x_train,y_train=y_train)

    return (x_data,y_data)


# 从目录中读取图片
def batch_extract_char_2_file(image_file_path=IN_PATH,out_to_folder=True,char_out_path = CHAR_OUT_PATH):
    '''
    # 从目录中读取图片，按图片名输出切分的字符到给定路径中
    :param image_file_path: 输入的图片的路径
    :param char_out_path: 输出路径
    :return: 
    '''
    img_file_names = os.listdir(image_file_path)
    page_list = []
    for file_name in img_file_names:
        img_path = image_file_path + os.path.sep + file_name
        if not os.path.isdir(img_path):
            img_org = io.imread(img_path, as_grey=True)
            img = np.where(img_org > 128, 1, 0)
            line_imgs = line_extract(img)
            path_name = os.path.splitext(file_name)[0]
            if out_to_folder:
                os.mkdir(char_out_path + os.path.sep + path_name)
            line_list = []
            for i, line_img in enumerate(line_imgs):
                char_position_arr = char_extract(line_img, [[0, line_img.shape[0], 0, line_img.shape[1]]])
                char_list = []
                for j, char_position in enumerate(char_position_arr[0]):
                    start_row, end_row, start_col, end_col = char_position
                    sub_img = line_img[start_row:end_row, start_col:end_col]
                    sub_img = np.where(sub_img == 1, 255, 0)
                    if (end_row-start_row) * (end_col - start_col) > 50:
                        if out_to_folder:
                            save_str = char_out_path + \
                                       os.path.sep + \
                                       path_name + \
                                       os.path.sep + \
                                       os.path.splitext(file_name)[0] + \
                                       "_" + str(i) + "_" + str(j) + ".png"
                            io.imsave(save_str, sub_img)
                        char_list.append(sub_img)
                line_list.append(char_list)
            page_list.append(line_list)

    return page_list,img_file_names


def classify_folder(path_in=CHAR_OUT_PATH,out_path=CLASSIFY_OUT_PATH):
    '''
    按照文件夹中的图片，对每个文件夹的图片进行分类
    :param path_in: 
    :param out_path: 
    :return: 
    '''
    file_names = os.listdir(path_in)
    for file_name in file_names:
        char_file_path = path_in + os.path.sep + file_name
        char_file_out_path = out_path + os.path.sep + file_name
        if os.path.exists(char_file_out_path):
            rmtree(char_file_out_path)
        os.mkdir(char_file_out_path)
        sample_img_list = classify_image_by_cosine_similarity(path_in=char_file_path,path_out=char_file_out_path)
        np.save(char_file_out_path+os.path.sep+file_name,sample_img_list)


def merge_classified_chars(path_in=CLASSIFY_OUT_PATH, path_out=MERGED_PATH):
    '''
    将已分类的文件夹合并
    :param path_in: 
    :param path_out: 
    :return: 
    '''
    path_in = r"D:\npq"
    path_out = MERGED_PATH
    sample_img_list_merged = None
    class_num = 0
    if os.path.exists(path_out):
        rmtree(path_out)
    total= len(os.listdir(path_in))
    for i,file_name in enumerate(os.listdir(path_in)):
        print("%d of %d" % (i, total))
        file_char_path = path_in + os.path.sep + file_name
        if i == 0:
            sample_img_list_merged = np.load(file_char_path+os.path.sep+file_name+".npy")
            class_num = sample_img_list_merged.shape[0]
            for char_class_path_name in os.listdir(file_char_path):
                input_folder_full_path = file_char_path + os.path.sep + char_class_path_name
                out_folder_full_path = path_out + os.path.sep + char_class_path_name
                if os.path.isdir(input_folder_full_path):
                    copytree(input_folder_full_path, out_folder_full_path)
        else:
            sample_img_list_new = np.load(file_char_path + os.path.sep + file_name + ".npy")
            for j, img_new in enumerate(sample_img_list_new):
                input_folder_full_path = file_char_path + os.path.sep + str(j)
                for k, img_merged in enumerate(sample_img_list_merged):
                    out_folder_full_path = path_out + os.path.sep + str(k)
                    if cosine_similarity_2(img_new,img_merged) > TH:
                        for char_file_name in os.listdir(input_folder_full_path):
                            copyfile(input_folder_full_path + os.path.sep + char_file_name,out_folder_full_path+ os.path.sep + char_file_name)
                        break
                else:
                    np.concatenate((sample_img_list_merged,[img_new]))
                    class_num += 1
                    out_folder_full_path = path_out + os.path.sep + str(class_num)
                    copytree(input_folder_full_path, out_folder_full_path)


def extract_top_n_sample(path_in=MERGED_PATH,path_out=SORTED_PATH,N = 500):
    dir_list = []
    for i, folder_name in enumerate(os.listdir(path_in)):
        folder_path = path_in + os.path.sep + folder_name
        num = len(os.listdir(folder_path))
        dir_list.append((int(folder_name),num))
    dir_list_new = sorted(dir_list,key=lambda x:x[1],reverse=True)
    arr = np.asarray(dir_list_new)
    arr_extract = arr[arr[:,1] > 2]
    if os.path.exists(path_out):
        rmtree(path_out)
    os.mkdir(path_out)
    for i, v in arr_extract:
        src = path_in + os.path.sep + str(i)
        dst = path_out + os.path.sep + str(i)
        copytree(src,dst)

# 使用已经训练好的编码器对样本进行编码
def encode_sample(encoder_file_path="encoder.h5",sample_file_path=r"D:\data.npz"):
    import keras
    x_train = np.load(sample_file_path)['x_train']
    x_train = np.reshape(x_train,(-1,60,60,1))
    encoder = keras.models.load_model(encoder_file_path)
    x_train_coded = encoder.predict(x_train)
    np.savez("D:/encoded_data.npz",x_train_coded)


def predict_image_with_consine_similarity(encoder_data_file_path=r"D:\encoded_data.npz",
                                  encoder_file_path="encoder.h5",
                                  rec_input_path=r"D:\Users\Riolu\Desktop\新建文件夹 (2)"):
    '''
    通过余弦相似度 标记 rec_input_path 中的数据
    :param encoder_data_file_path: 已经使用 encoder 编码的数据 
    :param encoder_file_path: encoder 模型文件
    :param rec_input_path: 识别的目录下的文件
    :return: 
    # TODO 将识别的文件归类
    '''
    import keras
    unrecognize_picture_dir = "D:\\img_out"
    x_train,y_train = np.load("D:/data.npz")['x_train'],np.load("D:/data.npz")['y_train']
    encoder = keras.models.load_model(encoder_file_path)
    t = len(os.listdir(rec_input_path))
    utfcode_str_list = []
    x_train_coded = np.load(encoder_data_file_path)['arr_0']
    for i,file_name in enumerate(os.listdir(rec_input_path)):
        full_path = rec_input_path + os.path.sep + file_name
        img = io.imread(full_path,as_grey=True)
        img_resize = transform.resize(util.invert(img),(60,60),mode='reflect')
        img2 = np.reshape(img_resize,(1,60,60,1))
        encoded_img = encoder.predict(img2)
        lbs = []
        for img_t in x_train_coded:
            lbs.append(cosine_similarity_2(encoded_img,img_t))
        mx = np.argmax(lbs)

        tibetan_word_code = []
        with open("words_Titan.txt") as f:
            for line in f.readlines():
                tibetan_word_code.append(line.strip())

        if lbs[mx] > 0.9:
            rec_str = tibetan_word_code[y_train[mx]]

            print(rec_str,file_name)
            utfcode_str_list.append(rec_str)
        else:
            utfcode_str_list.append('*')
            io.imsave(unrecognize_picture_dir+os.path.sep+file_name,img)
        # print("%d / %d" % (i,t))

    print(utfcode_str_list)


def predict_classified_image(encoder_data_file_path=r"D:\encoded_data.npz",
                                  encoder_file_path="encoder.h5",
                                  rec_input_path=r"D:\img_test",
                                  out_put_path = r"D:\img_out"
                             ):
    '''
    用于分类已经经过初步分类好的未进行人工标记的数据。由于模型已经训练好，此方法可以不用
    :param encoder_data_file_path: 
    :param encoder_file_path: 
    :param rec_input_path: 
    :param out_put_path: 
    :return: 
    '''
    labeled_data = np.load("D:/data.npz")
    x_train,y_train = labeled_data['x_train'],labeled_data['y_train']
    import keras
    encoder = keras.models.load_model(encoder_file_path)
    input_folders = os.listdir(rec_input_path)
    x_train_coded = np.load(encoder_data_file_path)['arr_0']
    tibetan_word_code = []
    with open("words_Titan.txt") as f:
        for line in f.readlines():
            tibetan_word_code.append(line.strip())

    for i,folder in enumerate(input_folders):
        img_folder_full_path = rec_input_path + os.sep + folder
        img_name = os.listdir(img_folder_full_path)[0]
        rec_img_path = rec_input_path + os.sep + folder + os.sep + img_name
        img = io.imread(rec_img_path,as_grey=True)
        img_resize = transform.resize(util.invert(img),(60,60),mode='reflect')
        img2 = np.reshape(img_resize,(1,60,60,1))
        encoded_img = encoder.predict(img2)
        lbs = []
        for img_t in x_train_coded:
            lbs.append(cosine_similarity_2(encoded_img,img_t))
        mx = np.argmax(lbs)

        if lbs[mx] > 0.9:
            out_classified_path = tibetan_word_code[y_train[mx]]
        else:
            out_classified_path = "unrecognized"

        for img_file in os.listdir(img_folder_full_path):
            img_full_path = os.path.join(img_folder_full_path, img_file)
            img_out_put_path = os.path.join(out_put_path, out_classified_path)
            if not os.path.exists(img_out_put_path):
                os.mkdir(img_out_put_path)
            copyfile(img_full_path, os.path.join(img_out_put_path, img_file))

        print("%d of %d" % (i,len(input_folders)))

def classifiy_chars_by_pages(path_in='/home/lyx2/img_in',path_out=r"/home/lyx2/img_out"):
    encoder_file_path = "./encoder.h5"
    encoder_data_file_path = r"./encoded_data.npz"
    labeled_data = np.load(r"./data.npz")
    temp_folder = r'./temp__'
    x_train, y_train = labeled_data['x_train'], labeled_data['y_train']
    if os.path.exists(temp_folder):
        rmtree(temp_folder)
    os.mkdir(temp_folder)
    batch_extract_char_2_file(char_out_path=temp_folder)
    import keras
    encoder = keras.models.load_model(encoder_file_path)
    x_train_coded = np.load(encoder_data_file_path)['arr_0']

    tibetan_word_code = []
    with open("words_Titan.txt") as f:
        for line in f.readlines():
            tibetan_word_code.append(line.strip())

    image_to_recog_list = os.listdir(temp_folder)
    for i,image_file_name in enumerate(image_to_recog_list):
        page_path = os.path.join(temp_folder,image_file_name)
        t = len(os.listdir(page_path))
        for j,img_name in enumerate(os.listdir(page_path)):
            img_path = os.path.join(page_path,img_name)
            charactor_img = io.imread(img_path,as_grey=True)
            img_resize = transform.resize(util.invert(charactor_img), (60, 60), mode='reflect')
            img2 = np.reshape(img_resize, (1, 60, 60, 1))
            encoded_img = encoder.predict(img2)
            lbs = []
            for img_t in x_train_coded:
                lbs.append(cosine_similarity_2(encoded_img, img_t))
            mx = np.argmax(lbs)

            if lbs[mx] > 0.9:
                out_classified_path = tibetan_word_code[y_train[mx]]
            else:
                out_classified_path = "unrecognized"

            img_out_put_path = os.path.join(path_out, out_classified_path)
            if not os.path.exists(img_out_put_path):
                os.mkdir(img_out_put_path)
            img_out_put_path = os.path.join(img_out_put_path,img_name)
            copyfile(img_path,img_out_put_path)

            print("Page %d, %d of %d" % (i,j,t))
    rmtree(temp_folder)

if __name__ == '__main__':
    extract_exp_data_by_path_name(path_in=r'D:\Users\Riolu\Desktop\merged_ultimate2017-06-08',wrap_size=100)
    # encode_sample()
    # label_image_by_path_name()
    # batch_extract_char_2_file()
    # tibetan_word_code = []
    # with open("words_Titan.txt") as f:
    #     for line in f.readlines():
    #         tibetan_word_code.append(line.strip())
    #
    # x_train,y_train = label_image_by_path_name("D:\labeled_data_4_exp",tibetan_word_code,True)
    #
    # img = x_train[0]
    # label = y_train[0]
    # plt.imshow(img,cmap='gray')
    # plt.title(tibetan_word_code[label])
    # plt.show()





















