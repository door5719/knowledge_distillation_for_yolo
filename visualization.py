import cv2
import numpy as np
import copy


def add_label(img, text, org, color, thickness):
    return cv2.putText(img=img,
                       text=text,
                       org=org,
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.5,
                       color=color,
                       thickness=thickness)


def visualize_persons(img, person_dicts):
    for person in person_dicts:
        bbox = copy.deepcopy(person['bbox_ltwh'])
        bbox[2] += (bbox[0] - 1)
        bbox[3] += (bbox[1] - 1)
        score = person['score']
        img = cv2.rectangle(img,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            (255, 0, 0), 2)
        img = add_label(img,
                        f'{score:.4f}',
                        (bbox[0], bbox[1] - 5),
                        (255, 0, 0), 2)
                        
    return img


def visualize_detection(img, output_dict, time):
    img = add_label(img, 'inference_time: {}ms'.format(
        int(time)), (0, 20), (255, 0, 0), 2)
    img = visualize_persons(img, output_dict['people'])
    cv2.imshow('detection', img)
    return img


def visualize_annotations(img, anns, color, alpha):
    origin_img = img.copy()
    for ann in anns:
        bbox = ann['bbox']
        cv2.rectangle(img,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1),
            color, 2)
        if 'score' in ann.keys():
            score = ann['score']
            img = add_label(img, '{:.4f}'.format(score),
                (bbox[0], bbox[1] - 25), color, 2)
    img = (img*alpha + origin_img*(1-alpha)).clip(0, 255).astype(np.uint8)
    return img

