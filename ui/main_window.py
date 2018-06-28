# coding:utf8
from __future__ import print_function, division

import sys

import keras.models
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle

from pic_process.char_recog import get_string_by_img
from pic_process.img_process import get_org_image, remove_noise, text_extract, line_extract, \
    char_extract, resize_img
from ui.Char_Edit import Char_Edit
from ui.Char_QDialog import Char_QDialog
from ui.Line_QDialog import Line_QDialog
from ui.QLabelExt import QLabelExt
from ui.auto_ui.window_main_ui import Ui_Window_Main

OPEN_FILE_NAME = "open_file_name"
DPI = 100 * 2
RISZIE_RATIO = 400
SELECTED_COLOR = "green"
DEFAULT_COLOR = "red"


# 2018.4.12 系统优化


class Main_Window(QMainWindow, Ui_Window_Main):
    # region 一些需要存储的变量的定义
    # 记录所处的阶段 0.刚读入图像 1.预处理阶段 2.行切分阶段 3.字符处理阶段 4.字符处理阶段
    processing_stage = 0
    # 该对象用于保存处理过程中各种中间图像 按下关闭文件按钮这个对象清空
    img_dic = {}
    # 用于存储当前画布中的图像的值
    show_img = None
    # 用于存储当前正在处理中的图像
    processing_img = None
    # 存储已经分为行的图片
    text_line_imgs = None
    # 最初的文本行信息
    text_line_position_array_org = None
    # 用于存储文本行文本行
    text_line_position_array = None
    # 文本行分割时候的行间留白 单位px
    text_line_margin = 30
    # 用于存储切分后的字符数组的位置，和text_line_image里面的image对应
    char_position_arr = None
    #
    layout_contents_top_margions = 10
    # axe_size
    axe_size = [0.0025, 0.0025, 0.995, 0.995]
    # char_rectangle 字符框的patch
    char_rectangle_patch = None
    # 识别的字符 label_list
    label_lists = []
    # 选中的文本行 用于对文本行的删除以及合并
    selected_text_line_index_list = []
    # 选中的字符 用于对字符进行合并，切分，删除等操作
    selected_char_index_list = []
    # 训练模型
    model = None

    # endregion

    def __init__(self, parent=None):
        # region 初始化窗体中部件和布局
        super(Main_Window, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("藏文历史文献识别系统")
        self.main_widget = QWidget(self)
        self.scroll = QScrollArea(self.main_widget)
        layout = QVBoxLayout(self.main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll)
        self.main_widget.setLayout(layout)
        # endregion

        # 提升点击响应速度，预先加载模型 2018-04-19
        model_path = 'tools\\model_lenet5_v2.h5'
        model_path = 'Tibetan_index\\model_lenet5_v2_transform.h5'
        # model_path = 'Tibetan_index\\model_lenet5_v2_extend.h5'
        model_path = 'Tibetan_index\\model_lenet5_v2_less_noise.h5'
        model_path = r'Tibetan_index\\model_lenet5_v2_TS.h5'
        model_path = r'Tibetan_index\\model_lenet5_v2_TS_2.h5'
        self.model = keras.models.load_model(model_path)

        # region 连接事件
        # 使用idaCode编码
        self.leaCode = self.read_leaCode()
        # 文件 menu
        # 打开文件按钮
        self.action_open.triggered.connect(self.action_open_file_triggered)
        # 关闭文件按钮
        self.action_close.triggered.connect(self.action_close_file_triggered)
        # 退出
        self.action_exit.triggered.connect(lambda: sys.exit(0))

        # 预处理 menu
        # 去除噪点
        self.action_remove_noise.triggered.connect(self.action_remove_noise_triggered)
        # 倾斜矫正
        # self.action_tilt_correction.triggered.connect(self.action_tilt_correction_triggered)
        # 检测文字区域
        self.action_text_extract.triggered.connect(self.action_text_extract_triggered)

        # 行切分
        # 切分
        self.action_line_extract.triggered.connect(self.action_line_extract_triggered)
        # 切分行
        self.action_cut_selected_line.triggered.connect(self.action_cut_selected_line_triggered)
        # 删除行
        self.action_del_selected_line.triggered.connect(self.action_del_selected_line_triggered)
        # 合并行
        self.action_merged_selected_line.triggered.connect(self.action_merge_selected_line_triggered)

        # 增大行间距
        # 缩小行间距

        # 字丁切分
        # 切分
        self.action_char_extract.triggered.connect(self.action_char_extract_triggered)
        # 删除字符
        self.action_del_selected_char.triggered.connect(self.action_del_selected_char_triggered)
        # 合并字符
        self.action_merge_selected_char.triggered.connect(self.action_merge_selected_char_triggered)
        # 手动切分相连的字符
        self.action_cut_selected_char.triggered.connect(self.action_cut_selected_char_triggered)

        # 识别已切分的字符
        self.action_recognize_chars.triggered.connect(self.action_recognize_chars_triggered)

        # 导出切分的字符到图片
        self.action_char_export_2_img.triggered.connect(self.action_char_export_2_img_triggered)
        # 导出切分的字符到文本
        self.action_char_export_2_text.triggered.connect(self.action_char_export_2_text_triggered)

        # 帮助
        # 关于
        self.action_about.triggered.connect(self.action_about_triggered)

        # endregion

        self.setCentralWidget(self.main_widget)

    # region function: 绘制 current_img 图像中的值
    def show_image_on_widget(self):
        show_img = self.show_img
        if not (show_img is None):
            main_widget = QWidget()
            main_layout = QVBoxLayout()
            if len(show_img.shape) == 3:
                rows, cols, channels = show_img.shape
            elif len(show_img.shape) == 2:
                rows, cols = show_img.shape
            figsize = cols / DPI, rows / DPI
            fig = plt.figure(figsize=figsize)
            self.ax = fig.add_axes(self.axe_size)
            self.ax.spines['left'].set_color('none')
            self.ax.spines['right'].set_color('none')
            self.ax.spines['bottom'].set_color('none')
            self.ax.spines['top'].set_color('none')
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.imshow(show_img, cmap="gray")
            self.canvas = FigureCanvas(fig)
            self.canvas.draw()
            self.show_img = show_img
            main_layout.addWidget(self.canvas)
            main_widget.setLayout(main_layout)
            self.scroll.setWidget(main_widget)
            plt.close()
        else:
            self.scroll.setWidget(QWidget())

    # endregion

    # region function: 绘制 行 将切分好的行图像拼接成 一张图像并展示文字区域
    def show_mutil_images_on_widget2(self):
        if self.text_line_imgs is None:
            self.scroll.setWidget(QWidget())
        else:
            main_widget = QWidget()
            main_layout = QVBoxLayout()
            # 生成需要展示的图像并更新行的位置
            rows, cols = self.processing_img.shape
            new_line_postion = []
            img = np.ones((self.text_line_margin, cols))
            row_count = img.shape[0]
            for i, line_img in enumerate(self.text_line_imgs):
                line_img_rows, line_img_cols = line_img.shape
                line_img_tmp = np.ones((line_img_rows, cols))
                line_img_tmp[:line_img_rows, :line_img_cols] = line_img
                img = np.vstack([img, line_img_tmp, np.ones((self.text_line_margin, cols))])
                self.text_line_imgs[i] = line_img_tmp
                new_line_postion.append((
                    row_count,
                    row_count + line_img_rows,
                    0,
                    cols))
                row_count += line_img.shape[0] + self.text_line_margin
            rows = img.shape[0]
            figsize = cols / RISZIE_RATIO * 2, rows / RISZIE_RATIO * 2
            fig = plt.figure(figsize=figsize)
            # print(figsize)
            self.ax = fig.add_axes(self.axe_size)
            self.ax.spines['left'].set_color('none')
            self.ax.spines['right'].set_color('none')
            self.ax.spines['bottom'].set_color('none')
            self.ax.spines['top'].set_color('none')
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.imshow(img, cmap='gray')
            rec_list = []
            for i, line_position in enumerate(new_line_postion):
                min_row, max_row, min_col, max_col = line_position
                rec = Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, fill=False, ec="red",
                                picker=True)
                rec_list.append(rec)
                rec.line_index = i
            for p in rec_list:
                self.ax.add_patch(p)
            self.canvas = FigureCanvas(fig)
            self.canvas.draw()
            self.canvas.mpl_connect('pick_event', self.on_line_pick)
            self.processing_img = img
            self.text_line_position_array = new_line_postion
            self.selected_text_line_index_list = []
            main_layout.addWidget(self.canvas)
            main_widget.setLayout(main_layout)
            self.scroll.setWidget(main_widget)

    # endregion

    # region  function: 绘制 字符 已经切分的字符数组
    def show_char_image_on_widget(self):

        if self.char_position_arr is None:
            return
        else:
            main_widget = QWidget()
            main_layout = QVBoxLayout()
            rows, cols = self.processing_img.shape
            figsize = cols / RISZIE_RATIO * 8, rows / RISZIE_RATIO * 8
            fig = plt.figure(figsize=figsize)
            self.ax = fig.add_axes(self.axe_size)
            self.ax.spines['left'].set_color('white')
            self.ax.spines['right'].set_color('white')
            self.ax.spines['bottom'].set_color('white')
            self.ax.spines['top'].set_color('white')
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.imshow(self.processing_img, cmap='gray')
            self.char_rectangle_patch = []
            for line_index, (line_char_postion, line_position) in enumerate(
                    zip(self.char_position_arr, self.text_line_position_array)):
                line_start_row, _, line_start_col, _ = line_position
                line_patch = []
                for char_index, char_position in enumerate(line_char_postion):

                    char_start_row, char_end_row, char_start_col, char_end_col = char_position
                    rect_x = char_start_col
                    rect_y = line_start_row + char_start_row
                    rect_width = char_end_col - char_start_col
                    rect_height = char_end_row - char_start_row
                    char_rect = Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, edgecolor="red",
                                          picker=True)
                    # 保存 char_rect 的 index
                    char_rect.char_index = [line_index, char_index]
                    line_patch.append(char_rect)

                self.char_rectangle_patch.append(line_patch)

            for line_patch in self.char_rectangle_patch:
                for p in line_patch:
                    self.ax.add_patch(p)

            self.canvas = FigureCanvas(fig)
            self.canvas.mpl_connect('pick_event', self.on_char_pick)
            main_layout.addWidget(self.canvas)
            self.canvas.draw()
            main_widget.setLayout(main_layout)
            self.scroll.setWidget(main_widget)
            self.selected_char_index_list = []
            plt.close()

    # endregion

    # region function: 打开文件
    def action_open_file_triggered(self):


        file_name_tuple = QFileDialog.getOpenFileName(self, "打开文件", ".", "图片文件 (*.jpg *.png)")
        file_name, file_name_filter = file_name_tuple

        if len(file_name) == 0:
            print("未选择任何文件")
        else:
            self.img_dic[OPEN_FILE_NAME] = file_name
            self.processing_img, self.show_img = get_org_image(file_name)
            self.show_image_on_widget()

    # endregion

    # region function: 关闭文件
    def action_close_file_triggered(self):
        self.show_img = None
        self.processing_img = None
        self.show_image_on_widget()

    # endregion

    # region function: 噪点去除
    def action_remove_noise_triggered(self):
        if self.show_img is None:
            return
        img = remove_noise(self.processing_img)
        self.show_img = resize_img(img)
        self.processing_img = img
        self.show_image_on_widget()

    # endregion

    # region function: 倾斜矫正
    # def action_tilt_correction_triggered(self):
    #     if self.show_img is None:
    #         return
    #     img = tilt_correction(self.processing_img)
    #     self.show_img = resize_img(img)
    #     self.processing_img = img
    #     self.show_image_on_widget()
    # endregion

    # ZQC 修正编码问题
    def read_leaCode(self):
        with open('./Tibetan_index/leaCode.txt') as readFile:
            charter = readFile.readline().replace('\n', '')
            s = set()
            while (len(charter) > 3):
                tem = chr(int(charter, 16))
                s.add(tem)
                charter = readFile.readline().replace('\n', '')
                # print(charter, "has been write")
        return s

    # region function: 文字区域检测
    def action_text_extract_triggered(self):
        if self.show_img is None:
            return
        img = text_extract(self.processing_img)
        self.show_img = resize_img(img)
        self.processing_img = img
        self.show_image_on_widget()
        self.processing_stage = 1

    # endregion

    # region function: 文本行切分
    def action_line_extract_triggered(self):

        if self.show_img is None:
            return
        if self.processing_stage == 1:
            imgs, line_position = line_extract(self.processing_img, with_line_position=True)
            self.text_line_position_array = line_position
            self.text_line_position_array_org = line_position
            self.text_line_imgs = imgs
            self.processing_stage = 2
            self.show_mutil_images_on_widget2()
            # self.processing_img = imgs

    # endregion

    # region function: 选中切分好的文本行的操作
    def on_line_pick(self, event):
        if event.mouseevent.button == 1:

            rec = event.artist
            line_index = rec.line_index
            if line_index not in self.selected_text_line_index_list:
                self.selected_text_line_index_list.append(line_index)
                rec.set_edgecolor(SELECTED_COLOR)
            else:
                self.selected_text_line_index_list.remove(line_index)
                rec.set_edgecolor(DEFAULT_COLOR)
            self.canvas.draw()

    # endregion

    # region function: 手动切分行
    def action_cut_selected_line_triggered(self):
        if self.processing_stage == 2:
            if len(self.selected_text_line_index_list) == 1:
                selected_line_id = self.selected_text_line_index_list[0]
                img = self.text_line_imgs[selected_line_id]
                rows, cols = img.shape
                line_dialog = Line_QDialog(self, img)
                line_dialog.setWindowTitle("手动行切分")
                ret_message = line_dialog.exec_()
                # print(ret_message)
                if ret_message == 1 and line_dialog.cut_position is not None:
                    cutting_position = int(np.floor(line_dialog.cut_position))
                    img2 = img[cutting_position:, :]
                    self.text_line_imgs[selected_line_id] = img[:cutting_position, :]
                    self.text_line_imgs.insert(selected_line_id + 1, img2)
                    self.show_mutil_images_on_widget2()
            elif len(self.selected_text_line_index_list) == 0:
                QMessageBox.information(self, "消息", "请选择一行进行切分")
            else:
                QMessageBox.information(self, "消息", "只能选择一行进行切分")

    # endregion

    # region function: 删除选中的文本行
    def action_del_selected_line_triggered(self):
        if self.processing_stage == 2:
            if len(self.selected_text_line_index_list) > 0:
                self.text_line_imgs = list(np.delete(self.text_line_imgs, self.selected_text_line_index_list, axis=0))
                self.selected_text_line_index_list = []
                self.show_mutil_images_on_widget2()

    # endregion

    # region function: 合并选中的文本行
    def action_merge_selected_line_triggered(self):
        if self.processing_stage == 2:
            if len(self.selected_text_line_index_list) >= 2:
                self.selected_text_line_index_list.sort()
                line_index_list = self.selected_text_line_index_list
                img = self.text_line_imgs[line_index_list[0]]
                rows, cols = img.shape
                if line_index_list[-1] - line_index_list[0] == len(line_index_list) - 1:
                    for index in line_index_list[1:]:
                        img = np.vstack(
                            [img, np.ones((self.text_line_margin, cols)), self.text_line_imgs[line_index_list[index]]])
                    self.text_line_imgs[line_index_list[0]] = img
                    self.text_line_imgs = list(
                        np.delete(self.text_line_imgs, self.selected_text_line_index_list[1:], axis=0))
                    self.show_mutil_images_on_widget2()
            else:
                QMessageBox.information(self, "提示", "只有连续的文本行才可以合并")

    # endregion

    # region function: 字符切分
    def action_char_extract_triggered(self):
        if self.processing_stage == 2:
            if self.show_img is None or self.text_line_position_array is None:
                return

            char_position_arr = char_extract(self.processing_img, self.text_line_position_array)

            self.char_position_arr = char_position_arr
            self.show_char_image_on_widget()
            self.processing_stage = 3

    # endregion

    # region function: 对选择的字符进行切分
    def action_cut_selected_char_triggered(self):
        if self.processing_stage == 3:
            if len(self.selected_char_index_list) == 1:
                line_index, char_index = self.selected_char_index_list[0]
                line_start_position = self.text_line_position_array[line_index][0]
                char_start_row, char_end_row, char_start_col, char_end_col = self.char_position_arr[line_index][
                    char_index]
                # 提取 选中的 bbox 中的内容，并在新的窗口中展示
                show_img = self.processing_img[char_start_row + line_start_position:char_end_row + line_start_position,
                           char_start_col:char_end_col]
                char_dialog = Char_QDialog(self, show_img)
                char_dialog.setWindowTitle("手动字丁切分")
                ret_message = char_dialog.exec_()
                # print(ret_message)
                if ret_message == 1 and char_dialog.cut_position is not None:
                    cutting_position = int(np.floor(char_dialog.cut_position))
                    print(cutting_position)
                    old_char_start_row, old_char_end_row, old_char_start_col, old_char_end_col = \
                        self.char_position_arr[line_index][char_index]
                    new_char_rect_1 = [old_char_start_row, old_char_end_row, old_char_start_col,
                                       old_char_start_col + cutting_position]
                    new_char_rect_2 = [old_char_start_row, old_char_end_row, old_char_start_col + cutting_position,
                                       old_char_end_col]
                    self.char_position_arr[line_index][char_index] = new_char_rect_1
                    # print("len",len(self.char_position_arr))
                    # del self.char_position_arr
                    if(len(self.char_position_arr) == 1):
                        total = list(self.char_position_arr)
                        tem_arry = list(self.char_position_arr[line_index])
                        tem_arry.insert(char_index+1,new_char_rect_2)
                        total.pop(line_index)
                        total.insert(line_index,tem_arry)
                        del self.char_position_arr
                        self.char_position_arr = np.asarray(total)
                    else:
                        self.char_position_arr[line_index] = np.insert(self.char_position_arr[line_index], char_index + 1,
                                                                   new_char_rect_2, axis=0)

                    char_dialog.close()
                    self.show_char_image_on_widget()
            else:
                QMessageBox.information(self, "消息", "只能选择一个字符进行切分")

    # endregion

    # region function: 删除选中的字符
    def action_del_selected_char_triggered(self):
        if self.processing_stage == 3:
            if len(self.selected_char_index_list) > 0:
                selected_char_list = self.selected_char_index_list
                selected_char_list.sort()
                last = None
                line_del_dict = {}
                for char_index in selected_char_list:
                    line_index, c_index = char_index
                    if last is None:
                        line_del_dict[line_index] = [c_index]
                        last = char_index
                        continue
                    else:
                        if line_index > last[0]:
                            line_del_dict[line_index] = [c_index]
                        else:
                            line_del_dict[line_index].append(c_index)
                        last = char_index
                print(line_del_dict)
                for i, v in line_del_dict.items():
                    if (len(self.char_position_arr) == 1):
                        total = list(self.char_position_arr)
                        tem_arry = list(self.char_position_arr[i])
                        tem_arry.pop(v[0])
                        total.pop(i)
                        total.insert(i, tem_arry)
                        del self.char_position_arr
                        self.char_position_arr = np.asarray(total)
                    else:
                        self.char_position_arr[i] = np.delete(self.char_position_arr[i], list(v), axis=0)
                self.show_char_image_on_widget()

    # endregion

    # region function: 合并选择的字符
    def action_merge_selected_char_triggered(self):
        if self.processing_stage == 3:
            if len(self.selected_char_index_list) > 0:
                selected_char_list = self.selected_char_index_list
                selected_char_list.sort()
                last = None
                line_del_dict = {}
                for char_index in selected_char_list:
                    line_index, c_index = char_index
                    if last is None:
                        line_del_dict[line_index] = [c_index]
                        last = char_index
                        continue
                    else:
                        if line_index > last[0]:
                            line_del_dict[line_index] = [c_index]
                        else:
                            line_del_dict[line_index].append(c_index)
                        last = char_index
                print(line_del_dict)
                if len(line_del_dict.values()) > 1:
                    QMessageBox.information(self, "提示", "请选择同一行的字符进行合并")
                else:
                    line_index, char_indexes = list(line_del_dict.items())[0]
                    to_merge_list = []
                    for char_id in char_indexes:
                        to_merge_list.append(self.char_position_arr[line_index][char_id])
                    min_row, _, min_col, _ = np.min(np.asarray(to_merge_list), axis=0)
                    _, max_row, _, max_col = np.max(np.asarray(to_merge_list), axis=0)
                    self.char_position_arr[line_index][char_indexes[0]] = [min_row, max_row, min_col, max_col]
                    if (len(self.char_position_arr) == 1):
                        total = list(self.char_position_arr)
                        tem_arry = list(self.char_position_arr[line_index])
                        for i in range(1,len(char_indexes),1):
                            tem_arry.pop(char_indexes[i])
                        total.pop(line_index)
                        total.insert(line_index, tem_arry)
                        del self.char_position_arr
                        self.char_position_arr = np.asarray(total)
                    else:
                        self.char_position_arr[line_index] = np.delete(self.char_position_arr[line_index], char_indexes[1:],
                                                                   axis=0)
                    print(min_row, max_row, min_col, max_col)
                    self.show_char_image_on_widget()

    # endregion

    # region function: 选中字符后展示手动切分的方法
    def on_char_pick(self, event):
        if event.mouseevent.button == 1:
            # 判断是否处于字切分阶段
            if self.processing_stage == 3:
                rec = event.artist
                line_index, char_index = rec.char_index
                rec_id = (line_index, char_index)
                if rec_id not in self.selected_char_index_list:
                    self.selected_char_index_list.append(rec_id)
                    rec.set_edgecolor(SELECTED_COLOR)
                else:
                    self.selected_char_index_list.remove(rec_id)
                    rec.set_edgecolor(DEFAULT_COLOR)
                self.canvas.draw()

    # endregion

    # region function:导出已切分的字符到指定文件夹下
    def action_char_export_2_img_triggered(self):
        if self.char_position_arr is None:
            return
        # 获取保存的文件夹
        file_name_tuple = QFileDialog.getExistingDirectory(self, "选择保存文件夹")

        if file_name_tuple == '':
            return
        print(file_name_tuple)
        for line_index, (line_char_position, line_position) in enumerate(
                zip(self.char_position_arr, self.text_line_position_array)):
            line_start_row, _, line_start_col, _ = line_position
            line_patch = []
            for char_index, char_position in enumerate(line_char_position):
                char_start_row, char_end_row, char_start_col, char_end_col = char_position
                char_start_row = line_start_row + char_start_row
                char_end_row = line_start_row + char_end_row
                img_sub = self.processing_img[char_start_row:char_end_row, char_start_col:char_end_col]
                saved_file_name = file_name_tuple + "\\line_%d_%d.png" % (line_index + 1, char_index)
                plt.imsave(saved_file_name, img_sub * 255, cmap="gray")
        QMessageBox.information(self, "提示", "字符已成功导出")

    # endregion

    # region function:导出已切分的字符到文本文件
    def action_char_export_2_text_triggered(self):
        if self.processing_stage == 4:
            text_file_name_tuple = QFileDialog.getSaveFileName(self, "选择保存的文件名", ".", "文本文件 (*.txt)")
            print(text_file_name_tuple)
            with open(text_file_name_tuple[0], 'w+', encoding="utf-8") as f:
                page_str = ''
                try:
                    for label_list in self.label_lists:
                        line_str = ''
                        for label in label_list:
                            line_str += str(label.text())
                            # line_str += str(label.name)
                        page_str += line_str + '\n'

                    f.write(page_str)
                except Exception as e:
                    print(e)
            QMessageBox.information(self, "提示", "字符已成功导出")

    # endregion

    # region function:按行展示出已识别的文字
    def action_recognize_chars_triggered(self):
        if self.processing_stage == 3:
            if self.char_position_arr is None or self.text_line_imgs is None:
                return
            Main_Widget = QWidget(self)
            Main_Widget_layout = QVBoxLayout(Main_Widget)
            Main_Widget_layout.setSizeConstraint(QLayout.SetNoConstraint)
            Main_Widget_layout.setContentsMargins(0, 0, 0, 0)
            line_char_number_list = [len(i) for i in self.char_position_arr]
            canvas_list = []
            line_chars_imgs = []
            for i, img in enumerate(self.text_line_imgs):
                # print(self.text_line_position_array[i])
                rows, cols = img.shape
                figsize = cols / DPI * 4, rows / DPI * 4
                fig = plt.figure(figsize=figsize)
                ax = fig.add_axes(self.axe_size)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(img, cmap="gray")
                canvas = FigureCanvas(fig)
                canvas.draw()
                for j, char_position in enumerate(self.char_position_arr[i]):
                    char_start_row, char_end_row, char_start_col, char_end_col = char_position
                    rect_x = char_start_col
                    rect_y = char_start_row
                    rect_width = char_end_col - char_start_col
                    rect_height = char_end_row - char_start_row
                    char_rect = Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, edgecolor="red",
                                          picker=True)
                    char_rect.char_index = (i, j)
                    ax.add_patch(char_rect)
                    line_chars_imgs.append(
                        img[char_start_row:char_end_row, char_start_col:char_end_col])
                # canvas.mpl_connect('pick_event', self.on_char_recog_pick)
                canvas_list.append(canvas)

                plt.close()

            unicodes = get_string_by_img(line_chars_imgs, model=self.model)
            txt_index = 0

            for i, canvas in enumerate(canvas_list):
                img = self.text_line_imgs[i]
                rows, cols = self.text_line_imgs[i].shape
                line_frame = QFrame(Main_Widget)
                # ToDO
                line_frame.setFixedSize(cols, rows * 4)
                line_text = unicodes[txt_index:txt_index + line_char_number_list[i]]
                label_font = QFont()
                label_font.setPointSize(20)
                label_list = []

                for j, reg_char in enumerate(line_text):
                    try:
                        min_row, max_row, min_col, max_col = self.char_position_arr[i][j]
                        img = self.text_line_imgs[i][min_row:max_row, min_col:max_col]
                        label = QLabelExt(img, line_frame)
                        if chr(int(reg_char[0], 16)) in self.leaCode:
                            label_font.setFamily("IeaUnicode")
                        else:
                            label_font.setFamily("IPAPANNEW")
                        label.setFont(label_font)
                        label.setText(chr(int(reg_char[0], 16)))
                        label.name = str(reg_char[0])
                        label.leacode = self.leaCode
                        label.char_id = i, j
                        label.setGeometry(QRect(min_col + 2, 30, 30, 30))
                        label_list.append(label)

                    except Exception as e:
                        print(e)

                if line_char_number_list[i] > 1:
                    Main_Widget_layout.addWidget(canvas)
                    Main_Widget_layout.addWidget(line_frame)
                    txt_index += line_char_number_list[i]
                    self.label_lists.append(label_list)

            Main_Widget.setLayout(Main_Widget_layout)
            self.scroll.setWidget(Main_Widget)
            self.processing_stage = 4

    # endregion

    # TODO 删除由图片产生的响应
    # region function:char选中识别的操作
    # def on_char_recog_pick(self, event):
    #     if event.mouseevent.button == 1:
    #         try:
    #             rec = event.artist
    #             line_index, char_index = rec.char_index
    #             start_row, end_row, start_col, end_col = self.char_position_arr[line_index][char_index]
    #             img = self.text_line_imgs[line_index][start_row:end_row, start_col:end_col]
    #             txt_label = self.label_lists[line_index][char_index]
    #             rows, cols = img.shape
    #             # print(txt_label.name)
    #             if chr(int(txt_label.name,16)) in self.leaCode:
    #                 message = str(txt_label.name)
    #             else:
    #                 message = str(txt_label.text())
    #             char_edit_dialog = Char_Edit(self, img, message)
    #
    #             # char_edit_dialog = Char_Edit(self, img, chr(int(txt_label.name,16)))
    #
    #             char_edit_dialog.setWindowTitle("修改识别错误的字丁")
    #             ret_message = char_edit_dialog.exec_()
    #             if ret_message == 1:
    #                 modified_str = char_edit_dialog.textEdit.toPlainText()
    #                 if len(modified_str) > 0:
    #                     txt_label.setText(modified_str)
    #
    #         except Exception as e:
    #             print(e)
    #
    # # endregion

    # region function:关于
    def action_about_triggered(self):
        txt = '''
        藏文历史文献数字化平台:
        作者：李颜兴
        感谢支持和帮助我工作的
        老师：马龙龙，段立娟
        同学：刘吉，江激扬，张西群，苏慧，刘泽宇
        '''
        QMessageBox.about(self, "关于", txt)

    # endregion
    pass
