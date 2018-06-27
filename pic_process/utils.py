# coding:utf-8
# 一些用到的工具类
from __future__ import print_function,division
import numpy as np

def get_line_range(profile,isTouching=False):
    '''
    根据profile找非0区域的起始值和结束值
    :param profile: 一维数组 
    :return:
    例如 
    >>> a = np.asarray([0,1,1,1,0,1,1,0,1,0])
    >>> print(get_line_range(a))
    [(1,4),(5,7),(9,10)]
    '''
    if (isTouching):
        MAXSIZE = 6
    else :
        MAXSIZE = 0
    start_index, end_index = 0, 0
    line_range = []
    cnt_flag = False
    for i, num in enumerate(profile):
        if not cnt_flag:
            if num > 0:
                cnt_flag = True
                start_index = i
        else:
            if num == 0:
                end_index = i
                cnt_flag = False
                if(end_index - start_index > MAXSIZE):
                    line_range.append((start_index, end_index))
                start_index, end_index = 0, 0
    # 最后一行加入到范围当中
    if cnt_flag:
        line_range.append((start_index, i))
    return line_range


def nmovlp(pnt1,pnt2):
    '''
    两个距离CC 的距离 left:l right:r top:t bottom:b 
    :param pnt1: t_1,l_1,b_1,r_1
    :param pnt2: t_2,l_2,b_2,r_2
    :return: 
    '''
    if pnt1[1] > pnt2[1]:
        pnt1,pnt2 = pnt2,pnt1
    t_1, l_1, b_1, r_1 = pnt1
    t_2, l_2, b_2, r_2 = pnt2
    dist =0.5 *abs(l_2+r_2-r_1-l_1)
    w_1 = r_1 - l_1
    w_2 = r_2 - l_2
    ovlp = w_2  if r_1 > r_2 else r_1 - l_2
    span = w_1  if r_1 > r_2 else r_2 - l_1
    if w_1 < w_2:
        ret = ovlp / w_1 - dist / span
    else:
        ret = ovlp / w_2 - dist / span
    return ret

# 计算两个像素块之间的距离 返回距离种类
def get_rectangle_distance(pnt1,pnt2):
    if pnt1[1] > pnt2[1]:
        pnt1,pnt2 = pnt2,pnt1

    l1, t1, r1, b1 = pnt1[0], pnt1[1], pnt1[2], pnt1[3]
    l2, t2, r2, b2 = pnt2[0], pnt2[1], pnt2[2], pnt2[3]

    t2mb1 = t2 - b1 # 1底边 到 2顶端 距离 2-bcde
    l1mr2 = l1 - r2 # 2右边 到 1左边 距离 01-a
    l2mr1 = l2 - r1 # 1右边 到 2左边 距离 01-f
    rtlb  = (l1mr2**2 + t2mb1**2)**.5  # 2-a
    ltrb  = (l2mr1**2 + t2mb1**2)**.5  # 2-a

    if l2 < l1: #A
        if r2 < l1: #a
            if t2 < b1: #2-a
                return '2-a',rtlb
            else: #01 -a
                if b2 < b1:
                    return '0-a',l1mr2
                else:
                    return '1-a',l1mr2
        elif l1 <= r2 <= r1: #b
            if t2 > b1: #2-b
                return '2-b',t2mb1
            else: #01-b
                if b2 < b1:
                    return '0-b',0
                else:
                    return '1-b',0
        else: #c
            if t2 > b1: #2-c
                return '2-c',t2mb1
            else:
                if b2 < b1:
                    return '0-c',0
                else:
                    return '1-c',0
    elif l1 <= l2 <= r1: #B
        if l1 <= r2 <= r1: #d
            if t2 > b1: #2-d
                return '2-d',t2mb1
            else: #01-d
                if b2 < b1:
                    return '0-d',0
                else:
                    return '1-d',0
        else: #e
            if t2 > b1: #2-e
                return '2-e',t2mb1
            else:
                if b2 < b1:
                    return '0-e',0
                else:
                    return '1-e',0
    else: #C
        if t2 < b1: #01-f
            if b2 < b1:
                return '0-f',l2mr1
            else:
                return '1-f',l2mr1
        else:
            return '2-f',ltrb

if __name__ == '__main__':
    pass
    # nmovlp([0L, 0L, 115L, 182L], [111L, 88L, 160L, 185L])