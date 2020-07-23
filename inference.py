#! /bin/python
# -*- coding: utf-8 -*-

"""
  * @author:zbl
  * @file: .py
  * @time: 2020/07/01
  * @func:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time, os
import json
import mmcv
import skimage.io
import colorsys
import numpy as np
import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector


zhfont1 = matplotlib.font_manager.FontProperties(size=11, fname="/anaconda3/envs/nlp-env/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/simhei.ttf")
DATASET_PATH = "/path/rubbish_classification_data/train"
with open(os.path.join(DATASET_PATH, 'annotations.json')) as f:
    json_file = json.load(f)


categories = json_file["categories"]
images = json_file['images']
annotations = json_file["annotations"]
categories_dict = dict()
categories_name2id_dict = dict()
for item in categories:
    categories_dict[item["id"]] = item["name"]
    categories_name2id_dict[item["name"]] = item["id"]
print(categories_dict)


def main(show=False):
    config_file = 'configs/cascade_rcnn_r50_fpn_1x_coco.py'  # 修改成自己的配置文件
    checkpoint_file = 'work_dirs/cascade_rcnn_r50_fpn_1x_coco/epoch_24.pth'  # 修改成自己的训练权重

    test_path = '/path/val/images'  # 官方测试集图片路径

    json_name = "result_" + "" + time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".json"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = []
    for img_name in os.listdir(test_path):
        if img_name.endswith('.png'):
            img_list.append(img_name)

    result = []
    for i, img_name in enumerate(img_list, 1):
        full_img = os.path.join(test_path, img_name)
        predict = inference_detector(model, full_img)
        single_img_bbox = []
        for i, bboxes in enumerate(predict, 1):
            if len(bboxes) > 0:
                defect_label = i
                # print(i)
                image_name = img_name
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2-x1, 2), round(y2-y1, 2)  # save 0.00
                    result.append({'name': image_name, 'category': defect_label-1, 'bbox': [x1, y1, x2, y2], 'score': score})
                    single_img_bbox.append([x1, y1, x2, y2, defect_label-1])

        if show:
            img = skimage.io.imread(test_path + f"/{img_name}")
            draw_img(img, single_img_bbox)

    with open(json_name, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def draw_img(image, bboxs, title="show"):
    height, width = image.shape[:2]
    _, ax = plt.subplots(1, figsize=(16, 16))
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    colors = random_colors(len(bboxs))
    masked_image = image.astype(np.uint32).copy()
    class_str = ""
    for i, bbox in enumerate(bboxs):
        x1, y1, w, h, label = bbox
        color = colors[i]
        p = patches.Rectangle((x1, y1), w, h, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)
        ax.text(x1, y1 + 15, f"{label}", color='w', size=15, backgroundcolor="black", fontproperties=zhfont1)
        class_str += f"{label} - {categories_dict[label]}\n"
    #     print(class_str)
    ax.text(10, len(bboxs) * 10, f"{class_str.strip()}", color='w', size=15, backgroundcolor="black",
            fontproperties=zhfont1)

    ax.imshow(masked_image.astype(np.uint8))
    plt.show()


if __name__ == "__main__":
    main(True)
