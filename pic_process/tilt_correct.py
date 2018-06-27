from __future__ import division,print_function

import os
import imghdr
import optparse

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks,rescale,rotate


def compare_sum(value):
    if value >= 44 and value <= 46:
        return True
    else:
        return False

def get_max_freq_elem(arr):
    max_arr = []
    freqs = {}
    for i in arr:
        if i in freqs:
            freqs[i] += 1
        else:
            freqs[i] = 1

    sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
    max_freq = freqs[sorted_keys[0]]

    for k in sorted_keys:
        if freqs[k] == max_freq:
            max_arr.append(k)

    return max_arr

def calculate_deviation(angle):
    angle_in_degrees = np.abs(angle)
    deviation = np.abs(np.pi/4 - angle_in_degrees)

    return deviation

def determine_skew(img_org):
    img = rescale(img_org,0.1)
    edges = canny(img, sigma=3.0)
    h, a, d = hough_line(edges)
    _, ap, _ = hough_line_peaks(h, a, d, num_peaks=20)

    absolute_deviations = [calculate_deviation(k) for k in ap]
    average_deviation = np.mean(np.rad2deg(absolute_deviations))
    ap_deg = [np.rad2deg(x) for x in ap]

    bin_0_45 = []
    bin_45_90 = []
    bin_0_45n = []
    bin_45_90n = []

    for ang in ap_deg:

        deviation_sum = int(90 - ang + average_deviation)
        if compare_sum(deviation_sum):
            bin_45_90.append(ang)
            continue

        deviation_sum = int(ang + average_deviation)
        if compare_sum(deviation_sum):
            bin_0_45.append(ang)
            continue

        deviation_sum = int(-ang + average_deviation)
        if compare_sum(deviation_sum):
            bin_0_45n.append(ang)
            continue

        deviation_sum = int(90 + ang + average_deviation)
        if compare_sum(deviation_sum):
            bin_45_90n.append(ang)

    angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
    lmax = 0

    for j in range(len(angles)):
        l = len(angles[j])
        if l > lmax:
            lmax = l
            maxi = j

    if lmax:
        ans_arr = get_max_freq_elem(angles[maxi])
        ans_res = np.mean(ans_arr)

    else:
        ans_arr = get_max_freq_elem(ap_deg)
        ans_res = np.mean(ans_arr)

    img2 = rotate(img_org,ans_res+90)
    return img2

if __name__ == '__main__':
    img = io.imread("D:/49.png",as_grey=True)
    img = determine_skew(img)
    img2 = rescale(img,0.3)
    plt.imshow(img2)
    plt.show()