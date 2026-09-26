import os
import cv2
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


'''
Main function - runs all others
'''

def changeNumbers(systolic, diastolic, pulse, color, PIL_digit_list, small_PIL_digit_list, reflection_list=None): # used to have im and corners as parameters
    # Change the background color
    background = colorBackground(color)
    # Add digits to the background
    new_background, boxes = add_specific_digits(background, systolic, diastolic, pulse, PIL_digit_list, small_PIL_digit_list)
    if reflection_list != None:
        r = random.randint(0,2)
        for _ in range(0, r):
            # Add reflections
            new_background = add_reflection(new_background, reflection_list)
    return new_background, boxes 


'''
Step 1: create random LCD color & create a background image with that color
'''

def colorBackground(color):


    # Choose random value for the color shade
    r = random.randint(0,120)


    if color == 'green':
        # Green: range (20,255,60)-(20,135,60)
        screen_color = [20,255-r,60]
        color_name = 'green'
    elif color == 'yellow':
        # Yellow: range (0,135,255)-(0,255,255)
        screen_color = [0,255-r,255]
        color_name = 'yellow'
    elif color == 'red':
        # Red: range (10,10,255)-(10,70,255)
        screen_color = [10,70-int(r/2),255]
        color_name = 'red'
    
    # Create a background with this color
    color = np.full((4800, 4300, 3), screen_color, np.uint8) 

    return color


'''
Step 2: add digits from digits folder on top of background image 

digits will be of size [1360, 770, 3] and have clear background
'''

#New

def add_box_xyxy(boxes, class_id, x1, y1, x2, y2):
    boxes.append({
        "class_id": class_id,
        "xyxy": [x1, y1, x2, y2]
    })

def add_specific_digits(
    background,
    systolic,
    diastolic,
    pulse,
    PIL_digit_list,
    small_PIL_digit_list
):

    IMG_W = 4300
    IMG_H = 4800

    BIG_W = 700
    BIG_H = 1400

    SMALL_W = 360
    SMALL_H = 600

    boxes = []

    s_systolic = str(systolic)
    s_diastolic = str(diastolic)
    s_hr = str(pulse)

    PIL_background = Image.fromarray(background)

    # ------------------------
    # SBP digits
    # ------------------------
    sbp_y = 500
    sbp_digit_positions = []

    if len(s_systolic) == 3:
        sbp_digit_positions.append((s_systolic[0], 1400, sbp_y))

    sbp_digit_positions.append((s_systolic[-2], 2250, sbp_y))
    sbp_digit_positions.append((s_systolic[-1], 3100, sbp_y))

    for digit, x, y in sbp_digit_positions:
        add_digits_to_location(PIL_background, PIL_digit_list, digit, x, y)
        add_box_xyxy(boxes, int(digit), x, y, x + BIG_W, y + BIG_H)

    sbp_x1 = min(x for _, x, _ in sbp_digit_positions)
    sbp_y1 = sbp_y
    sbp_x2 = 3100 + BIG_W
    sbp_y2 = sbp_y + BIG_H

    add_box_xyxy(boxes, 10, sbp_x1, sbp_y1, sbp_x2, sbp_y2)

    # ------------------------
    # DBP digits
    # ------------------------
    dbp_y = 2000
    dbp_digit_positions = []

    if len(s_diastolic) == 3:
        dbp_digit_positions.append((s_diastolic[0], 1400, dbp_y))

    dbp_digit_positions.append((s_diastolic[-2], 2250, dbp_y))
    dbp_digit_positions.append((s_diastolic[-1], 3100, dbp_y))

    for digit, x, y in dbp_digit_positions:
        add_digits_to_location(PIL_background, PIL_digit_list, digit, x, y)
        add_box_xyxy(boxes, int(digit), x, y, x + BIG_W, y + BIG_H)

    dbp_x1 = min(x for _, x, _ in dbp_digit_positions)
    dbp_y1 = dbp_y
    dbp_x2 = 3100 + BIG_W
    dbp_y2 = dbp_y + BIG_H

    add_box_xyxy(boxes, 10, dbp_x1, dbp_y1, dbp_x2, dbp_y2)

    # ------------------------
    # Pulse digits
    # ------------------------
    pulse_y = 3700
    pulse_digit_positions = []

    if len(s_hr) == 3:
        pulse_digit_positions.append((s_hr[0], 2750, pulse_y))

    pulse_digit_positions.append((s_hr[-2], 3150, pulse_y))
    pulse_digit_positions.append((s_hr[-1], 3550, pulse_y))

    for digit, x, y in pulse_digit_positions:
        add_digits_to_location(PIL_background, small_PIL_digit_list, digit, x, y)
        add_box_xyxy(boxes, int(digit), x, y, x + SMALL_W, y + SMALL_H)

    # optional symbols, not annotated
    add_digits_to_location(PIL_background, small_PIL_digit_list, "heart", 2500, 3800)
    add_digits_to_location(PIL_background, small_PIL_digit_list, "mmHg", 3150, 3400)

    pulse_x1 = min(x for _, x, _ in pulse_digit_positions)
    pulse_y1 = pulse_y
    pulse_x2 = 3550 + SMALL_W
    pulse_y2 = pulse_y + SMALL_H

    add_box_xyxy(boxes, 10, pulse_x1, pulse_y1, pulse_x2, pulse_y2)

    background = np.array(PIL_background)

    return background, boxes

'''
Helper function to Step 2
'''

def add_digits_to_location(PIL_background, PIL_digit_list, s_dig, x, y):
    # Read in the image from the digit list
    digit = PIL_digit_list[s_dig]
    # Paste the image on top of the background
    PIL_background.paste(digit, (x,y), mask=digit)



'''
Step 3: Add reflections
'''

def add_reflection(screen, reflection_list):

    r = random.randint(1,len(reflection_list))
    rotation_amount = random.randint(-30, 30)

    reflection = reflection_list[r-1]

    image_center = tuple(np.array(reflection.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rotation_amount, 1.1) # zooms in slightly to avoid black background
    reflection = cv2.warpAffine(reflection, rot_mat, reflection.shape[1::-1], flags=cv2.INTER_LINEAR)

    reflection = cv2.resize(reflection, (screen.shape[1], screen.shape[0]))

    alpha = random.uniform(0.08, 0.18)
    screen = cv2.addWeighted(screen, 1, reflection, alpha, 0)

    return screen