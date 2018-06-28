# coding:utf8
from __future__ import print_function, division

from threading import Thread,RLock
from multiprocessing import Queue,Pool
mylock = RLock()

from skimage import data, io, filters, color, measure
from skimage.measure import label, regionprops
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks, rescale,probabilistic_hough_line,rotate

from pic_process.tilt_correct import determine_skew
from pic_process.utils import get_line_range, get_rectangle_distance, nmovlp
from matplotlib.patches import Rectangle
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# region 一些经验参数常量
# 噪点大小，认为
NOISE_SIZE_THRESHOLD = 15
RESIZE_WIDTH = 900
# 8 邻域联通区域
CONNECTIVITY = 8
# 文本方向
TEXTLINEDIRECTION = 1
# 提取文本区域预留的留白(margins)
TEXT_MARGIN = 80
# 竖直投影时后规定的噪声的大小
V_NOISE = 40
# 背景颜色定义 白色1 黑色0
BACKGROUND_VALUE = 1

# 解决轻微粘连问题，小于该值，认定为一个分割点
PROJECTION_THRESHOLD = 5

# 存储在垂直投影过程中出现误合并的图像,默认存储
isSave = True
tem_path = r'./dd'

# 设定是否是模糊粘连字符
isTouching = False
# endregion

# region function: 获取图像
def get_org_image(file_name):
    img = io.imread(file_name)
    if len(img.shape) > 2:
        img = color.rgb2gray(img)
    else:
        img = img / 255
    img = np.where(img > 0.99, img, 0)
    rows, cols = img.shape
    resize_ratio = RESIZE_WIDTH / cols
    resize_img = rescale(img, resize_ratio)
    return img,resize_img


# endregion

# region function: 去除噪点
def remove_noise(img):
    # 噪声分为两类，一类是小的像素点，一类是和边框连在一起的
    rows, cols = img.shape
    # plt.imshow(img)
    # plt.show()
    img_bin = np.where(img >= 0.99, 0, 1)
    img_labeled = label(img_bin)

    regions = regionprops(img_labeled)

    for region in regions:
        region_label = region.label
        minr, minc, maxr, maxc = region.bbox
        # 和边框连接在一起的噪声去除
        if minr == 0 or minc == 0 or maxr >= rows or maxc == cols or region.area < NOISE_SIZE_THRESHOLD:
            for point in region.coords:
                row,col = point
                img[row,col] = 1
    return img


# endregion

# region function: 倾斜矫正
def tilt_correction(img):
    img_org = img
    img = determine_skew(img)
    return img
# endregion

# region function: 文字区域提取
def text_extract(img):
    # 水平h_profile 竖直v_profile 投影 投影时前景色1背景色0
    h_profile = np.sum(np.where(img>=0.99,0,1),axis=1)
    v_profile = np.sum(np.where(img>=0.99,0,1),axis=0)
    # 获取 水平 竖直 非文本区域的范围，计算均值，去除小于均值/2里的内容
    h_range = get_line_range(h_profile)
    v_range = get_line_range(v_profile)

    h_range_arr = np.array(h_range)
    v_range_arr = np.array(v_range)

    h_range_value = h_range_arr[:,1] - h_range_arr[:,0]
    v_range_value = v_range_arr[:,1] - v_range_arr[:,0]

    avg_h = np.average(h_range_value)
    avg_v = np.average(v_range_value)

    valid_rows = h_range_arr[np.where(h_range_value > avg_h/2,True,False)]
    start_row,end_row = valid_rows[0][0],valid_rows[-1][-1]
    valid_cols = v_range_arr[np.where(v_range_value > avg_v/2,True,False)]
    start_col,end_col = valid_cols[0][0],valid_cols[-1][-1]

    print(img[start_row:end_row,start_col:end_col].shape)

    return img[start_row:end_row,start_col:end_col]
# endregion 文字区域提取

# region function: 文本行提取
def line_extract(img, with_line_position=False):
    # 提取的个数和范围
    h_profile = np.sum(np.where(img==1,0,1),axis=1)
    h_range = get_line_range(h_profile)
    # 新建一个数组用于存储提取图像的范围
    text_line_range_arr = []
    imgs_out = []
    rows,cols = img.shape
    for index, (start_row,end_row) in enumerate(h_range):
        v_profile = np.sum(np.where(img[start_row:end_row,:]==1,0,1),axis=0)
        # 处理到此步骤认为该输入图像已经去除了噪点
        v_range = get_line_range(v_profile)
        start_col = v_range[0][0]
        end_col = v_range[-1][-1]
        text_line_range_arr.append([start_row,end_row,start_col,start_col+cols])
        imgs_out.append(img[start_row:end_row,start_col:start_col+cols])
    if with_line_position:
        return imgs_out,text_line_range_arr
    else:
        return imgs_out
# endregion

char_position_arr_g = []
class LineSegmentWorker(Thread):

    def __init__(self,line_index,line_position):
        self.line_index = line_index
        self.line_position = line_position
        pass

    def run(self,possessing_img):
        line_start_row,line_end_row,line_start_col,line_end_col = self.line_position
        line_img = possessing_img[line_start_row:line_end_row,:]
        print("line: %d" % self.line_index)
        v_profile = np.sum(np.where(line_img >= 0.99, 0, 1), axis=0)
        v_ranges = get_line_range(v_profile)
        global char_position_arr_g
        if len(v_ranges) == 1:
            mylock.acquire()
            char_position_arr_g.append([1,[line_start_row,line_start_row,line_start_row,line_start_row]])
            mylock.release()
            return
        char_line_arr = []
        v_arr = np.asarray(v_ranges)
        v_range_width = (v_arr[:,1]-v_arr[:,0])
        pass
def alter_num(nums):
    flag = True
    for i in range (len(nums)):
        if(nums[i] <= 5 and flag):
            nums[i] = 0
            flag = False
        elif(not flag) and (nums[i] > 5):
            flag = True
    return nums
# region function: 文字切分
def char_extract(possessing_img,line_position_array):
    # 传入的数据是经过切割后的文本行
    # 文字切分后的数组表示是行内的相对位置，在绘制bonding box的时候需要将纵坐标加上行的起始位置
    char_position_arr = []
    for line_index,line_position in enumerate(line_position_array):
        line_start_row,line_end_row,line_start_col,line_end_col = line_position
        line_img = possessing_img[line_start_row:line_end_row,:]
        print("line: %d" % line_index)
        v_profile = np.sum(np.where(line_img >= 0.99, 0, 1), axis=0)
        AREA = 10
        MAX = 0

        if(isTouching):
            v_profile = alter_num(v_profile)
            AREA = 20
            MAX = 5
        # v_profile = np.where(v_profile > 5, v_profile, 0)
        v_ranges = get_line_range(v_profile, isTouching)
        if len(v_ranges) == 1:
            tem = []
            tem.append([0,line_end_row - line_start_row,0,v_ranges[0][1] - v_ranges[0][0]])
            # tem.append([0,0,0,0])
            char_position_arr.append(tem)
            continue
        char_line_arr = []
        v_arr = np.asarray(v_ranges)
        v_range_width = (v_arr[:,1]-v_arr[:,0])

        # 使用kmeans方法聚类，计算，较大块儿的平均值
        width_kmeans = KMeans(n_clusters=2)
        width_kmeans.fit(v_range_width.reshape((-1,1)))
        avg1 = np.average(v_range_width[width_kmeans.labels_.astype(np.bool)])
        avg2 = np.average(v_range_width[np.bitwise_not(width_kmeans.labels_.astype(np.bool))])
        avg = avg1 if avg1 > avg2 else avg2

        for index, (char_start_col, char_end_col) in enumerate(v_ranges):
            h_profile = np.sum(np.where(line_img[:,char_start_col:char_end_col]==1,0,1),axis=1)
            h_range = get_line_range(h_profile)

            if len(h_range) > 0:
                char_start_row,char_end_row = h_range[0][0],h_range[-1][-1]
                # char_line_arr.append(img[start_row:end_row,start_col:end_col])
                # 使用联通区域分析法，来检测错分的字符
                img_sub = line_img[char_start_row:char_end_row, char_start_col:char_end_col]
                img_black = np.where(img_sub >= 0.99, 0, 1)
                if char_end_col - char_start_col > avg / 2:
                    label_sub_img = label(img_black,neighbors=4)
                    regions = regionprops(label_sub_img)

                    if isSave:
                        import os
                        file_path = os.path.join(tem_path, 'lb_'+str(line_index) + '_' + str(index) +'.png')
                        io.imsave(file_path, img_black*255)
                    # print(len(regions))
                    if len(regions) < 2:
                        char_line_arr.append([char_start_row, char_end_row, char_start_col, char_end_col])

                    else:
                        # if line_index == 1:
                        #     plt.imshow(img_black)
                        #     plt.show()
                        region_range_list = []
                        for region in regions:
                            if (region.area > AREA) and (region.bbox[3] - region.bbox[1] > MAX) and (region.bbox[2] - region.bbox[0] > MAX) :
                                region_range_list.append([region.bbox[0],region.bbox[1],region.bbox[2],region.bbox[3]])
                        region_range_list = sorted(region_range_list, key=lambda x: x[1])

                        region_label = [1] * len(region_range_list)
                        region_to_del = []
                        # region_merged_id = []
                        region_merge_list = []
                        # if region.area < 20:
                        #     region_to_del.append(region)
                        #     continue
                        for i in range(len(region_range_list)-1):
                            region = region_range_list[i]
                            to_merged_region_set = set()
                            if region_label[i] == 0:
                                continue
                            for j,region_to_cmp in enumerate(region_range_list[i+1:],i+1):
                                if region_label[i] == 0:
                                    continue
                                else:
                                    degree = nmovlp(region,region_to_cmp)
                                    to_merged_region_set.add(i)
                                    if degree > 0.2:
                                        to_merged_region_set.add(j)
                                        region_label[j] = 0

                            region_merge_list.append(to_merged_region_set)

                        if region_label[len(region_range_list)-1] == 1:
                            region_merge_list.append({len(region_range_list)-1})

                        new_region_list = []
                        for to_merged_regions in region_merge_list:
                            merged_region_min_row = 1000
                            merged_region_min_col = 1000
                            merged_region_max_row = 0
                            merged_region_max_col = 0
                            for mg_idx in to_merged_regions:
                                merged_region_min_row = min(merged_region_min_row, region_range_list[mg_idx][0])
                                merged_region_min_col = min(merged_region_min_col, region_range_list[mg_idx][1])
                                merged_region_max_row = max(merged_region_max_row, region_range_list[mg_idx][2])
                                merged_region_max_col = max(merged_region_max_col, region_range_list[mg_idx][3])
                            new_region_list.append([merged_region_min_row, merged_region_min_col,
                                                    merged_region_max_row, merged_region_max_col])
                        # print(new_region_list)

                        region_range_list = new_region_list

                        for region in region_range_list:
                            bbox_range = list(map(lambda x, y: x + y, [char_start_row, char_start_row, char_start_col, char_start_col],
                                             [region[0],region[2],region[1],region[3]]))
                            char_line_arr.append(bbox_range)

                else:
                    char_line_arr.append([char_start_row,char_end_row,char_start_col,char_end_col])
        char_position_arr.append(char_line_arr)

    return np.asarray(char_position_arr)

# region function by CC designed by ZQC
def char_extract_by_CC(possessing_img,line_position_array):
    char_position_arr = []
    for line_index, line_position in enumerate(line_position_array):
        line_start_row, line_end_row, line_start_col, line_end_col = line_position
        line_img = possessing_img[line_start_row:line_end_row, :].astype(np.int32)
        lb_image = measure.label(line_img, connectivity=1)
        regions = measure.regionprops(lb_image)
        count = 0
        rows, cols = line_img.shape
        totallist = []
        for region in regions:
            totallist.append([region.bbox, region.image.astype(np.int32), 0, region.area])
        for i in range(len(totallist)):
            for j in range((len(totallist))):
                confidence_value = nmovlp(totallist[i][0], totallist[j][0])
                if totallist[i][0][1] < totallist[j][0][1] and totallist[i][0][3] < totallist[j][0][3]:
                    min_x1, mid_x1, mid_x2, max_x2 = totallist[i][0][1], totallist[j][0][1], totallist[i][0][3], \
                                                     totallist[j][0][3]
                elif (totallist[i][0][1] > totallist[j][0][1] and totallist[i][0][3] > totallist[j][0][3]):
                    min_x1, mid_x1, mid_x2, max_x2 = totallist[j][0][1], totallist[i][0][1], totallist[j][0][3], \
                                                     totallist[i][0][3]
                else:
                    min_x1, mid_x1, mid_x2, max_x2 = np.sort(
                        [totallist[j][0][1], totallist[i][0][1], totallist[j][0][3], totallist[i][0][3]])
                if totallist[i][2] == 2:
                    break
                if totallist[j][2] == 2:
                    continue
                if i == j:
                    continue
                if (confidence_value > 0.3):
                    print(totallist[i][0], totallist[j][0], "confidence", confidence_value)
                    if (totallist[j][2] == 1):
                        totallist[j][2] == 2
                    if (totallist[i][0][1] < totallist[j][0][1] and totallist[i][0][3] > totallist[j][0][3]) or (
                            totallist[j][0][1] < totallist[i][0][1] and totallist[j][0][3] > totallist[i][0][3]):
                        min_x1, mid_x1, mid_x2, max_x2 = np.sort(
                            [totallist[i][0][1], totallist[j][0][1], totallist[i][0][3], totallist[j][0][3]])
                        min_y1, a, b, max_y2 = np.sort(
                            [totallist[i][0][0], totallist[j][0][0], totallist[i][0][2], totallist[j][0][2]])
                        a1_y1, a1_y2, a1_x1, a1_x2 = totallist[i][0][1] - min_x1, totallist[i][0][3] - min_x1, \
                                                     totallist[i][0][
                                                         0] - min_y1, totallist[i][0][2] - min_y1
                        b1_y1, b1_y2, b1_x1, b1_x2 = totallist[j][0][1] - min_x1, totallist[j][0][3] - min_x1, \
                                                     totallist[j][0][
                                                         0] - min_y1, totallist[j][0][2] - min_y1
                        tem_image1 = np.zeros((max_y2 - min_y1, max_x2 - min_x1))
                        tem_image2 = np.zeros((max_y2 - min_y1, max_x2 - min_x1))

                        row, col = totallist[i][1].shape
                        for ii in range(row):
                            for jj in range(col):
                                tem_image1[a1_x1 + ii][a1_y1 + jj] = totallist[i][1][ii][jj]
                        row, col = totallist[j][1].shape
                        for iii in range(row):
                            for jjj in range(col):
                                tem_image2[b1_x1 + iii][b1_y1 + jjj] = totallist[j][1][iii][jjj]

                        tem_image = tem_image1 + tem_image2
                        tem = np.where(tem_image > 0, 1, 0)
                        # print(tem_image.shape)
                        totallist[i][1] = tem
                        totallist[i][0] = [min_y1, min_x1, max_y2, max_x2]
                        totallist[j][2] = 2
                        totallist[i][3] = totallist[i][3] + totallist[j][3]
                    else:
                        min_y1, a, b, max_y2 = np.sort(
                            [totallist[i][0][0], totallist[j][0][0], totallist[i][0][2], totallist[j][0][2]])
                        a1_y1, a1_y2, a1_x1, a1_x2 = totallist[i][0][1] - min_x1, totallist[i][0][3] - min_x1, \
                                                     totallist[i][0][
                                                         0] - min_y1, totallist[i][0][2] - min_y1
                        b1_y1, b1_y2, b1_x1, b1_x2 = totallist[j][0][1] - min_x1, totallist[j][0][3] - min_x1, \
                                                     totallist[j][0][
                                                         0] - min_y1, totallist[j][0][2] - min_y1
                        tem_image1 = np.zeros((max_y2 - min_y1, max_x2 - min_x1))
                        tem_image2 = np.zeros((max_y2 - min_y1, max_x2 - min_x1))
                        tem_image = np.zeros((max_y2 - min_y1, max_x2 - min_x1))
                        # dd = plt.figure()
                        # dd.add_axes([0, 0, 1, 1]).imshow(totallist[i][1])
                        # plt.show()

                        tem_image1[a1_x1:a1_x2, a1_y1:a1_y2] = totallist[i][1]
                        tem_image2[b1_x1:b1_x2, b1_y1:b1_y2] = totallist[j][1]
                        tem_image = tem_image1 + tem_image2
                        tem = np.where(tem_image > 0, 1, 0)
                        # print(tem_image.shape)
                        totallist[i][1] = tem
                        totallist[i][0] = [min_y1, min_x1, max_y2, max_x2]
                        totallist[j][2] = 2
                        totallist[i][3] = totallist[i][3] + totallist[j][3]
        # 保存粘连字丁串
        # for i in range(len(totallist)):
        #     if(totallist[i][2] != 2):
        #         min_row, min_col, max_row, max_col = totallist[i][0][0], totallist[i][0][1], totallist[i][0][2],totallist[i][0][3]
        #         if  (max_col - min_col) <= 1.3 * count:
        #             io.imsave("D:\\datebase\\character\\isolate_character\\" + file.split(".")[0] + str(i) + ".png",totallist[i][1] * 255, cmap="gray")
        #             print("D:\\datebase\\character\\isolate_character\\" + file.split(".")[0] + str(i) + ".png")
        #         elif((max_col - min_col) > 1.3 * count and (max_col - min_col) <= 1.7 * count):
        #            io.imsave("D:\\datebase\\character\\touch_character\\two_touch_character\\" + file.split(".")[0] + str(i) + ".png", totallist[i][1] * 255, cmap="gray")
        #            print("D:\\datebase\\character\\touch_character\\two_touch_character\\" + file.split(".")[0] + str(i) + ".png")
        #         else:
        #             io.imsave("D:\\datebase\\character\\touch_character\\three_touch_character\\" + file.split(".")[0] + str(i) + ".png", totallist[i][1] * 255, cmap="gray")
        #             print("D:\\datebase\\character\\touch_character\\three_touch_character\\" + file.split(".")[0] + str(i) + ".png")

        # 处理元音符号
        touch_character = np.zeros((rows, cols)).astype(np.int32)
        temtt = np.zeros((rows, cols)).astype(np.int32)
        totalBBox = []
        for i in range(len(totallist)):
            if (totallist[i][2] != 2):
                min_row, min_col, max_row, max_col = totallist[i][0][0], totallist[i][0][1], totallist[i][0][2], \
                                                     totallist[i][0][3]
                if (max_col - min_col) > 2 * count:
                    temtt[min_row:max_row, min_col:max_col] = totallist[i][1]
                    touch_character = touch_character + temtt
                    tem = [min_row, min_col, max_row, max_col]
                    totalBBox.append(tem)

        # sss = plt.subplot(111)
        # sss.imshow(touch_character)
        # print("ddd", count)
        # plt.show()
        touch_character = np.where(touch_character > 0, 1, 0)
    return touch_character, totalBBox

# region function: 重新调整图像的大小到宽度为1200像素
def resize_img(img):
    rows, cols = img.shape
    resize_ratio = RESIZE_WIDTH / cols
    img = rescale(img, resize_ratio)
    return img
# endregion

# region main: 测试方法用的主函数
if __name__ == '__main__':
    img2 = io.imread(r"D:\1.png")
    img2 = color.rgb2gray(img2)
    # img = titl_correction2(img2)
    img = remove_noise(img2)
    img = text_extract(img)
    imgs_out,line_position = line_extract(img,with_line_position=True)
    rows, cols = img.shape
    resize_ratio = RESIZE_WIDTH / cols
    char_arr = char_extract(img,line_position)
    rows, cols = img.shape
    figsize = cols / 200, rows / 200
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.025, 0.025, 0.95, 0.95])
    ax.imshow(img, cmap="gray")
    char_position = char_extract(img,line_position)
    for line_range in line_position:
        line_start_row, line_end_row, line_start_col, line_end_col = line_range
        line_rect = Rectangle((line_start_col,line_start_row),line_end_col- line_start_col,line_end_row - line_start_row,fill=False,edgecolor='blue')
        ax.add_patch(line_rect)

    for line_range,chars in zip(line_position,char_arr):
        line_start_row,line_start_col = line_range[0],line_range[2]
        for charactor in chars:
            start_row, end_row, start_col, end_col = charactor
            char_rect = Rectangle((start_col, start_row+line_start_row), end_col - start_col, end_row - start_row, fill=False,
                                  ec="red")
            ax.add_patch(char_rect)
    plt.show()
# endregion

pass
