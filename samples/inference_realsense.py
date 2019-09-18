# coding: utf-8
import os
import sys
import time
import numpy as np
import cv2
from realsensecv import RealsenseCapture

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
# model.load_weights(COCO_MODEL_PATH, by_name=True)
# Load weights trained on MSCOCO2017add
model_path = model.find_last()
model.load_weights(model_path, by_name=True)
print('weight loaded!')

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

filtered_classNames = ['BG', 'bottle', 'cup', 'banana', 'orange', 'remote', 'book', 'cell phone']


def calc_center(img):
    '''重心座標(x,y)を求める'''
    mu = cv2.moments(img, False)
    x = int(mu["m10"] / (mu["m00"] + 1e-7))
    y = int(mu["m01"] / (mu["m00"] + 1e-7))
    return x, y


cap = RealsenseCapture()
cap.WIDTH = 1280
cap.HEIGHT = 960
cap.start()

while True:
    start_time = time.time()
    ret, images = cap.read()
    rgb_image = images[0]
    depth_image = images[1]
    # Run detection
    results = model.detect([rgb_image], verbose=1)
    # Visualize results
    result = results[0]

    N = result['rois'].shape[0]  # 検出数
    result_image = rgb_image.copy()
    colors = visualize.random_colors(N)

    for i in range(N):
        '''クラス関係なく1物体ごと処理を行う'''
        if class_names[result['class_ids'][i]] in filtered_classNames:
            # Color
            color = colors[i]
            rgb = (round(color[0] * 255), round(color[1] * 255), round(color[2] * 255))
            # Bbox
            result_image = visualize.draw_box(result_image, result['rois'][i], rgb)
            # Class & Score
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_top = class_names[result['class_ids'][i]] + ':' + str(result['scores'][i])
            result_image = cv2.putText(result_image, text_top,
                                       (result['rois'][i][1], result['rois'][i][0]),
                                       font, 0.8, rgb, 2, cv2.LINE_AA)
            # print(class_names[r['class_ids'][i]])
            # Mask
            mask = result['masks'][:, :, i]
            result_image = visualize.apply_mask(result_image, mask, color)
            # calculate distance
            mask_binary = mask.astype('uint8')
            center_pos = calc_center(mask_binary)
            distance = cap.depth_frame.get_distance(center_pos[0], center_pos[1])
            text_bottom = 'distance: {:.3f}m'.format(distance)
            result_image = cv2.putText(result_image, text_bottom,
                                       (result['rois'][i][1], result['rois'][i][2]),
                                       font, 0.8, rgb, 2, cv2.LINE_AA)
            print('distance: ', distance)

    # ヒートマップに変換
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
        depth_image, alpha=0.08), cv2.COLORMAP_JET)

    cv2.imshow('Mask R-CNN', np.hstack((result_image, depth_colormap)))
    print('FPS:', 1 / (time.time() - start_time))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
