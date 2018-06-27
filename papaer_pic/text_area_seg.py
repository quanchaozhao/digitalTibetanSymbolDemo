# coding:utf-8

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

img = io.imread("tt.png", as_grey=True)
# 图片的大小

rows,cols = img.shape
h_profile = np.sum(np.where(img >= 0.99, 0, 1), axis=1) / 7
v_profile = np.sum(np.where(img >= 0.99, 0, 1), axis=0) / 7


# DPI设为100时候，即为原图的大小
DPI = 600

# 根据图像的真实大小计算figsize
figsize = cols/ DPI ,rows / DPI
fig = plt.figure(figsize=figsize)
# 使用 [0,0,1,1] 屏蔽边框以及将图像填满整个区域
ax = fig.add_axes([0, 0, 1, 1])
# 起始坐标290,341 宽度84，高度46，笔画宽度为3

# 注意一定要先绘制图，然后再绘制Rectangle
ax.set_frame_on(False)
ax.imshow(img,cmap="gray")
ax.fill_between(h_profile,0,np.arange(len(h_profile)),color="blue",alpha=0.7)
ax.fill_between(np.arange(len(v_profile)),0,v_profile,color="blue",alpha=0.7)
# plt.show()
# 保存绘图到文件当中 .jpg .png .tif .svg
fig.savefig("test.png")