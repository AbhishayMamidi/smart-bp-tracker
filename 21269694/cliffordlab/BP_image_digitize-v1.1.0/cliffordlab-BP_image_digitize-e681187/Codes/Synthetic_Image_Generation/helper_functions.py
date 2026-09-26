import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image


def adjust_gamma(image, gamma=1.0):
        """
        Credit: https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python
        Parameters:
            image: A grayscale image (NxM int array in [0, 255]
            gamma: A positive float. If gamma<1 the image is darken / if gamma>1 the image is enlighten / if gamma=1 nothing happens.
        Returns: the enlighten/darken version of image
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
        return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


def get_lcd_yolo_rgb(image):
    if image is None:
        return None

    image = cv2.resize(image, (int(0.4 * 480), int(0.4 * 640)))

    # Ensure uint8 BGR input
    img = image.astype(np.uint8)

    # Illumination correction (apply per-channel via LUT if your adjust_gamma supports it)
    img = adjust_gamma(img, gamma=0.7)

    # Edge-preserving smoothing (works best with uint8)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Soft sharpening (use float32 temporarily)
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), sigmaX=1.2)
    sharp = cv2.addWeighted(f, 1.3, blur, -0.3, 0)

    return np.clip(sharp, 0, 255).astype(np.uint8)


'''
Paste synthetic LCD image into the empty iHealth device screen.
'''

def paste_lcd_into_device(lcd_img, device_template_path):
    
    """
    lcd_img: synthetic screen image as OpenCV image
    device_template_path: path to empty iHealth image
    """

    device = cv2.imread(device_template_path)

    h_lcd, w_lcd = lcd_img.shape[:2]

    # Approximate four corners of the empty screen in your iHealth image
    # Order: top-left, top-right, bottom-right, bottom-left

    dst_pts = np.float32([
        [254,243],
        [813,243],
        [813,925],
        [254,925]
    ])

    src_pts = np.float32([
        [0, 0],
        [w_lcd, 0],
        [w_lcd, h_lcd],
        [0, h_lcd]
    ])

    # Perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_lcd = cv2.warpPerspective(
        lcd_img,
        M,
        (device.shape[1], device.shape[0])
    )

    # Create mask from warped LCD
    mask = cv2.warpPerspective(
        np.ones((h_lcd, w_lcd), dtype=np.uint8) * 255,
        M,
        (device.shape[1], device.shape[0])
    )

    result = device.copy()
    screen_region = mask > 0

    result[screen_region] = warped_lcd[screen_region]

    return result, M



REALISTIC_LCD_COLORS = {
    "green": [
        (110, 255, 80),
        (90, 245, 70),
        (120, 240, 100),
        (140, 255, 120),
    ],

    "red": [
        (40, 40, 255),
        (20, 30, 230),
        (50, 70, 255),
        (30, 60, 220),
    ],

    "yellow": [
        (0, 210, 255),
        (30, 190, 240),
        (10, 180, 220),
    ]
}


def realistic_lcd_colorize(lcd, color):
    base_color = np.array(random.choice(REALISTIC_LCD_COLORS[color]), dtype=np.float32)

    gray = cv2.cvtColor(lcd, cv2.COLOR_BGR2GRAY)

    # dark digit mask
    digit_mask = gray < 80

    # create colored LCD background
    bg = np.ones_like(lcd, dtype=np.float32)
    bg[:, :] = base_color

    # keep digits dark
    out = bg.copy()
    out[digit_mask] = lcd[digit_mask]

    return np.clip(out, 0, 255).astype(np.uint8)


def add_lcd_gradient(img):
    h, w = img.shape[:2]

    x = np.linspace(
        random.uniform(0.75, 0.90),
        random.uniform(1.00, 1.15),
        w
    )

    gradient = np.tile(x, (h, 1)).astype(np.float32)
    gradient = cv2.GaussianBlur(gradient, (101, 101), 0)

    out = img.astype(np.float32)

    for c in range(3):
        out[:, :, c] *= gradient

    return np.clip(out, 0, 255).astype(np.uint8)


def add_lcd_noise(img):
    noise = np.random.normal(0, random.uniform(2, 6), img.shape).astype(np.int16)

    out = np.clip(
        img.astype(np.int16) + noise,
        0,
        255
    )

    return out.astype(np.uint8)


def make_lcd_realistic(lcd, color):
    lcd = realistic_lcd_colorize(lcd, color)
    lcd = add_lcd_gradient(lcd)
    lcd = add_lcd_noise(lcd)

    alpha = random.uniform(0.88, 1.12)
    beta = random.randint(-8, 8)

    lcd = cv2.convertScaleAbs(lcd, alpha=alpha, beta=beta)

    return lcd



def transform_boxes_with_matrix(boxes, M, out_w, out_h):
    transformed_boxes = []

    for box in boxes:
        class_id = box["class_id"]
        x1, y1, x2, y2 = box["xyxy"]

        corners = np.float32([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]).reshape(-1, 1, 2)

        warped = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

        xs = warped[:, 0]
        ys = warped[:, 1]

        new_x1 = np.clip(xs.min(), 0, out_w - 1)
        new_y1 = np.clip(ys.min(), 0, out_h - 1)
        new_x2 = np.clip(xs.max(), 0, out_w - 1)
        new_y2 = np.clip(ys.max(), 0, out_h - 1)

        # keep only boxes that still have visible area
        if (new_x2 - new_x1) > 2 and (new_y2 - new_y1) > 2:
            transformed_boxes.append({
                "class_id": class_id,
                "xyxy": [
                    float(new_x1),
                    float(new_y1),
                    float(new_x2),
                    float(new_y2)
                ]
            })

    return transformed_boxes


def boxes_xyxy_to_yolo(boxes, img_w, img_h):
    yolo_boxes = []

    for box in boxes:
        class_id = box["class_id"]
        x1, y1, x2, y2 = box["xyxy"]

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        yolo_boxes.append([
            class_id,
            x_center,
            y_center,
            width,
            height
        ])

    return yolo_boxes


def random_transform_device_with_boxes(img, boxes):
    transform_type = random.choice([
        "normal",
        "rotate_only",
        "tilt_only",
        "rotate_and_tilt"
    ])

    out = img.copy()
    out_boxes = boxes.copy()

    h, w = out.shape[:2]

    if transform_type in ["rotate_only", "rotate_and_tilt"]:
        out, R = rotate_device_with_matrix(out)
        out_boxes = transform_boxes_with_matrix(out_boxes, R, out_w=w, out_h=h)

    if transform_type in ["tilt_only", "rotate_and_tilt"]:
        out, P = tilt_device_with_matrix(out)
        out_boxes = transform_boxes_with_matrix(out_boxes, P, out_w=w, out_h=h)

    return out, out_boxes, transform_type

def rotate_device_with_matrix(img):
    h, w = img.shape[:2]
    angle = random.uniform(-12, 12)

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Convert 2x3 affine matrix to 3x3 perspective-style matrix
    M_3x3 = np.vstack([M, [0, 0, 1]])

    return rotated, M_3x3

def tilt_device_with_matrix(img):
    h, w = img.shape[:2]

    margin_x = random.randint(5, 20)
    margin_y = random.randint(3, 15)

    src = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    dst = np.float32([
        [random.randint(0, margin_x), random.randint(0, margin_y)],
        [w - random.randint(0, margin_x), random.randint(0, margin_y)],
        [w - random.randint(0, margin_x), h - random.randint(0, margin_y)],
        [random.randint(0, margin_x), h - random.randint(0, margin_y)]
    ])

    P = cv2.getPerspectiveTransform(src, dst)

    tilted = cv2.warpPerspective(
        img,
        P,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return tilted, P   