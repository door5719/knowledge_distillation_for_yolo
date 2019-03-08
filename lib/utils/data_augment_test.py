"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

Ellis Brown, Max deGroot
"""

import cv2
import numpy as np
from data_augment import draw_bbox,_crop,_distort,_elastic,_expand,_mirror,_rotate

if __name__ == '__main__':
    image = cv2.imread('/data/codes/yolov3/experiments/person.jpg')
    boxes = np.array([np.array([165, 100, 290, 380])]) # xmin, ymin, xmax, ymax
    labels = np.array([[1]])
    p = 1
    print(image.shape)
    image_show = draw_bbox(image, boxes)
    cv2.imshow('input_image', image_show)
    '''
    image_t, boxes, labels = _crop(image, boxes, labels)
    image_show = draw_bbox(image_t, boxes)
    cv2.imshow('crop_image', image_show)
    
    image_t = _distort(image_t)
    image_show = draw_bbox(image_t, boxes)
    cv2.imshow('distort_image', image_show)

    image_t = _elastic(image_t, p)
    image_show = draw_bbox(image_t, boxes)
    cv2.imshow('elastic_image', image_show)

    image_t, boxes = _expand(image_t, boxes, (103.94, 116.78, 123.68), p)
    image_show = draw_bbox(image_t, boxes)
    cv2.imshow('expand_image', image_show)

    image_t, boxes = _mirror(image_t, boxes)
    image_show = draw_bbox(image_t, boxes)
    cv2.imshow('mirror_image', image_show)
    '''

    image_t, boxes = _rotate(image, boxes)
    print(image_t.shape)
    image_show = draw_bbox(image_t, boxes)
    cv2.imshow('rotate_image', image_show)

    cv2.waitKey(100000)
    