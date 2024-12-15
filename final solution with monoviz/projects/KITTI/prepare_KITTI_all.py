import json
import numpy as np
import shutil
import os
import cv2

def mkdir_if_missing(directory, delete_if_exist=False):
    """
    Recursively make a directory structure even if missing.

    if delete_if_exist=True then we will delete it first
    which can be useful when better control over initialization is needed.
    """
    if delete_if_exist and os.path.exists(directory):
        shutil.rmtree(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

def read_calib(calib_path):
    with open(calib_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 2:
                line = line.strip()
                calib = np.array(line.split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)
                return calib
    return None

cats = ['Car', 'Van', 'Truck', 'Tram', 'Pedestrian', 'Cyclist', 'Person_sitting', 'Misc', 'DontCare']
cat2id = {cat: i for i, cat in enumerate(cats)}
cat_info = [{'name': cat, 'id': i} for i, cat in enumerate(cats)]

home_path = '.'
split = 'val1'  # This is just a directory name
label_dir = os.path.join(home_path, 'training', 'label_2')
calib_dir = os.path.join(home_path, 'training', 'calib')
image_dir = os.path.join(home_path, 'training', 'image_2')

# We use only the 'train' subset now and do not split data further.
subset = 'val'
ret = {'images': [], 'annotations': [], "categories": cat_info}

calib_dir_des = os.path.join(home_path, split, subset, 'calib')
image_dir_des = os.path.join(home_path, split, subset, 'image_2')
label_dir_des = os.path.join(home_path, split, subset, 'label_2')

mkdir_if_missing(image_dir_des)
mkdir_if_missing(label_dir_des)
mkdir_if_missing(calib_dir_des)

image_id = -1
# The input file (train.txt) should contain all image indices you want to include
for i in range(7481):
    line = f"{i:06d}"
    image_id += 1
    if image_id % 500 == 0:
        print(split, subset, image_id)

    calib_path = os.path.join(calib_dir, line + '.txt')
    calib = read_calib(calib_path)
    image_path = os.path.join(image_dir, line + '.png')
    label_path = os.path.join(label_dir, line + '.txt')

    # Keep the original file names (no renaming)
    image_path_des = os.path.join(image_dir_des, line + '.png')
    calib_path_des = os.path.join(calib_dir_des, line + '.txt')
    label_path_des = os.path.join(label_dir_des, line + '.txt')

    shutil.copy(image_path, image_path_des)
    shutil.copy(calib_path, calib_path_des)
    shutil.copy(label_path, label_path_des)

    image = cv2.imread(image_path)
    image_height = image.shape[0]
    image_width = image.shape[1]

    image_info = {
        'file_name': line + '.png',     # original file name
        'id': image_id,                 # internal ID can remain
        'calib': calib.tolist(),
        'height': image_height,
        'width': image_width
    }
    ret['images'].append(image_info)

    # Since all are considered training images, parse annotations
    with open(label_path, 'r') as anns:
        for ann_ind, txt in enumerate(anns):
            txt = txt.strip()
            tmp = txt.split(' ')
            cat_id = cat2id[tmp[0]]
            truncated = float(tmp[1])
            occluded = int(tmp[2])
            alpha = float(tmp[3])
            bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
            dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
            location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
            rotation_y = float(tmp[14])

            ann = {
                'image_id': image_id,
                'id': len(ret['annotations']),
                'category_id': cat_id,
                'truncated': truncated,
                'occluded': occluded,
                'alpha': alpha,
                'bbox': bbox,
                'dimensions': dim,
                'location': location,
                'rotation_y': rotation_y
            }
            ret['annotations'].append(ann)


print("### image num: ", len(ret['images']))
print("### annotation num: ", len(ret['annotations']))

json_path = os.path.join(home_path, split, 'KITTI_' + split + '_' + subset + '.json')
with open(json_path, 'w') as f:
    json.dump(ret, f)

