import os
import cv2
import sys
import random
import shutil
import numpy as np
import PIL.ImageOps
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt



'''
Pixelate the image by a random amount
'''

def pixelate(im):

    PADDING = 10

    r = random.randint(1, 15)

    y,x,_ = im.shape

    im = cv2.resize(cv2.resize(im, (int((x)/r)+PADDING, int((y)/r)+PADDING)), (x,y))

    return im


'''
Gausian blur
'''

def gaussian_blur(im):
    #im_g = im.copy()
    PADDING = 10

    r = random.randint(0, 30)

    if r%2==0:
        r +=1

    im = cv2.GaussianBlur(im, (r + PADDING, r + PADDING), 0)

    return im


'''
Rotate image & adjust background
'''
def rotate(im):
    #im_r = im.copy()

    r = random.randint(-5, 5)


    image_center = tuple(np.array(im.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, r, 1.1) # zooms in slightly to avoid black background
    im = cv2.warpAffine(im, rot_mat, im.shape[1::-1], flags=cv2.INTER_LINEAR)


    return im


'''
Squishes the rows
'''
def squish_rows(im):

    r = random.uniform(0.9, 1)


    rows, cols, ch = im.shape    

    org_points = np.float32(
        [[0, rows*(1-((1-r)/2))],
        [cols, rows*(1-((1-r)/2))],
        [0, rows*((1-r)/2)],
        [cols, rows*((1-r)/2)]])
    new_points = np.float32(
        [[0, rows],
        [cols, rows],
        [0, 0],
        [cols, 0]])

    # calculate the perspective transform matrix and warp the perspective
    M = cv2.getPerspectiveTransform(org_points, new_points)
    im = cv2.warpPerspective(im, M, (cols,rows))

    return im


'''
Squishes the columns
'''
def squish_columns(im):
    r = random.uniform(0.9, 1)


    rows, cols, ch = im.shape    

    org_points = np.float32(
        [[cols*((1-r)/2), rows], # 0, rows
        [cols*(1-((1-r)/2)), rows], # cols, rows
        [cols*((1-r)/2), 0], # 0, 0
        [cols*(1-((1-r)/2)), 0]]) # cols, 0
    new_points = np.float32(
        [[0, rows],
        [cols, rows],
        [0, 0],
        [cols, 0]])

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(org_points, new_points)
    im = cv2.warpPerspective(im, M, (cols,rows))

    return im




'''
Random contrast
'''
def random_contrast(im):
    #im_c = im.copy()

    r = random.uniform(0.8, 1.2)

    contrast=float(r)
    im = cv2.addWeighted(im, contrast, im, 0, 1)

    return im




'''
Random brightness
'''
def random_brightness(im):
    #im_b = im.copy()

    r = random.uniform(0,30)

    brightness=float(r) #100
    im = cv2.addWeighted(im, 1, im, 0, brightness)

    return im



