import pandas as pd
from edit_image import *
from changing_numbers import *
from helper_functions import *


NUM_REFLECTIONS = 6 

# Inclusive ranges
SBP_START = 60 # SBP start of range
SBP_END = 240 # SBP end of range

DBP_START = 30 # DBP start of range
DBP_END = 180 # DBP end of range

HR_START = 25
HR_END = 200 



path = 'path/to/directory/to/save/generated/images'
template_path = "./add_to_screen/iHealth_Empty.png"  # or your real path

dataset_dir = os.path.join(path, 'generated_BP_with_Frame_Labels')

train_img_dir = os.path.join(dataset_dir, "train", "images")
train_lbl_dir = os.path.join(dataset_dir, "train", "labels")
valid_img_dir = os.path.join(dataset_dir, "valid", "images")
valid_lbl_dir = os.path.join(dataset_dir, "valid", "labels")

for ddir in [train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir]:
    os.makedirs(ddir, exist_ok=True)


# Store metadata here
records = []

# Reading in all the digits
PIL_digit_list = dict() # For the SBP/DBP readings
small_PIL_digit_list = dict() # For the HR
for i in range(10):
    dig = Image.open('add_to_screen/markings/' + str(i) + '.png')
    big_dig = dig.resize((700, 1400))
    small_dig = dig.resize((360, 600))
    PIL_digit_list[str(i)] = big_dig
    small_PIL_digit_list[str(i)] = small_dig

# Reading in symbols
small_PIL_digit_list['heart'] = Image.open('add_to_screen/markings/heart.png')
small_PIL_digit_list['mmHg'] = Image.open('add_to_screen/markings/mmHg.png')

# Reading in reflections
reflection_list = list()
for i in range(1, NUM_REFLECTIONS+1):
    reflection = cv2.imread(f'add_to_screen/reflections/reflection{i}.png')
    reflection_list.append(reflection)


REPEATS_PER_VALUE = 5

required_sbps = {s: REPEATS_PER_VALUE for s in range(SBP_START, SBP_END + 1)}
required_dbps = {d: REPEATS_PER_VALUE for d in range(DBP_START, DBP_END + 1)}
required_pulses = {p: REPEATS_PER_VALUE for p in range(HR_START, HR_END + 1)}

samples = []

while required_sbps or required_dbps or required_pulses:

    if required_sbps:
        s = random.choice(list(required_sbps.keys()))
        required_sbps[s] -= 1
        if required_sbps[s] == 0:
            del required_sbps[s]
    else:
        s = random.randint(SBP_START, SBP_END)

    min_dbp = max(DBP_START, s - 50)
    max_dbp = min(DBP_END, s - 30)

    if min_dbp > max_dbp:
        continue

    possible_required_dbps = [
        d for d in required_dbps.keys()
        if min_dbp <= d <= max_dbp
    ]

    if possible_required_dbps:
        d = random.choice(possible_required_dbps)
        required_dbps[d] -= 1
        if required_dbps[d] == 0:
            del required_dbps[d]
    else:
        d = random.randint(min_dbp, max_dbp)

    if required_pulses:
        p = random.choice(list(required_pulses.keys()))
        required_pulses[p] -= 1
        if required_pulses[p] == 0:
            del required_pulses[p]
    else:
        p = random.randint(HR_START, HR_END)

    samples.append((s, d, p))


valid_ratio = 0.20
n_valid = int(len(samples) * valid_ratio)
valid_indices = set(random.sample(range(len(samples)), n_valid))

# while i < NUM_IMAGES:
for i, (s, d, p) in enumerate(samples):
    
    # Conditions based on screen colors of iHealth device
    if (s <= 129 and d <= 84):
        color = 'green'
    elif (((s <= 159 and s >= 130) or (d <= 99 and d >= 85)) and (s <= 159 and d <= 99)):
        color = 'yellow'
    else:
        color = 'red'
    

    # Generate LCD
    lcd, lcd_boxes = changeNumbers(
        s, d, p,
        color,
        PIL_digit_list,
        small_PIL_digit_list,
        reflection_list=None
    )
    
    # LCD-only effects, but NO rotation/squish
    lcd = pixelate(lcd)
    lcd = get_lcd_yolo_rgb(lcd)
    
    # make LCD color more realistic
    lcd = make_lcd_realistic(lcd, color)

    if random.random() < 0.7:
        lcd = add_reflection(lcd, reflection_list)
    
    # small contrast/brightness variation after realistic coloring
    lcd = random_contrast(lcd)
    lcd = random_brightness(lcd)

    # Use larger LCD size for better quality
    bp_img = cv2.resize(lcd, (260, 300), interpolation=cv2.INTER_CUBIC)

    
    ORIG_W = 4300
    ORIG_H = 4800
    scale_x = 260 / ORIG_W
    scale_y = 300 / ORIG_H

    resized_boxes = []

    for box in lcd_boxes:
    
        x1, y1, x2, y2 = box["xyxy"]
        resized_boxes.append({
            "class_id": box["class_id"],
            "xyxy": [
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ]
        })
    

    # Paste LCD into device
    final_img, M = paste_lcd_into_device(
        lcd_img=bp_img,
        device_template_path=template_path
    )


    h_final, w_final = final_img.shape[:2]
    
    final_boxes_xyxy = transform_boxes_with_matrix(
        resized_boxes,
        M,
        out_w=w_final,
        out_h=h_final
    )

    final_img, final_boxes_xyxy, transform_type = random_transform_device_with_boxes(
        final_img,
        final_boxes_xyxy
    )

    final_boxes = boxes_xyxy_to_yolo(
        final_boxes_xyxy,
        img_w=w_final,
        img_h=h_final
    )

    # Optional gentle blur only
    if random.random() < 0.3:
        final_img = cv2.GaussianBlur(final_img, (3, 3), 0)

    # Save image
    filename = (
        f"SBP{s}_DBP{d}_Pulse{p}_{color}_{i}.jpg"
    )


    # Decide train or valid
    if i in valid_indices:
        img_save_dir = valid_img_dir
        lbl_save_dir = valid_lbl_dir
        split = "valid"
    else:
        img_save_dir = train_img_dir
        lbl_save_dir = train_lbl_dir
        split = "train"
    
    # Save image
    image_path = os.path.join(img_save_dir, filename)
    cv2.imwrite(image_path, final_img)
    
    # Save YOLO label file
    label_filename = filename.replace(".jpg", ".txt")
    label_path = os.path.join(lbl_save_dir, label_filename)
    
    with open(label_path, "w") as f:
        for box in final_boxes:
            class_id, x_center, y_center, width, height = box
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


    records.append({
        "Filename": filename,
        "SBP": s,
        "DBP": d,
        "Pulse": p,
        "Color": color,
        "Transform": transform_type
    })


df = pd.DataFrame(records)

csv_path = os.path.join(dataset_dir, "Synthetic_BP_Labels.csv")
df.to_csv(csv_path, index=False)

print(f"\nSaved CSV to:\n{csv_path}")