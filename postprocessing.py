import cv2

import numpy as np

def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Either multiple boxes given as a 2D ndarray or single box as 1D ndarray
        return np.hstack((xyxy[..., 0:2], xyxy[..., 2:4] - xyxy[..., 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')



def postprocess(output_blobs, scale, img_shape):
    """Transform model output into the standard API format

    Args:
        output_blobs (dict): model output containing all detection info
        scale: image scale
        img_shape: image dimensions

    Returns:
        dict, with the following keys:
            'people': tuple of people detected
            'carts': tuple of carts detected
    """
    classes = ['background', 'head']
    index = 0
    people = []
    for class_ind in range(1, 2):
        class_name = classes[class_ind]
        if class_name == 'head':
            for object_ind in range(len(output_blobs['cls_boxes'])):
                entry = {
                    'score': 0,
                    'bbox_ltwh': np.zeros(4, dtype=np.int32),
                    'keypoints': np.zeros((1, 2), dtype=np.int32),
                    'keypoints_score': np.zeros((1, 1), dtype=np.float32),
                    'face_feature': np.zeros(512, dtype=np.float32),
                    'face_score': 0.,
                    'face_angle': 0.
                }
                score = float(output_blobs['scores'][index])
                bbox = output_blobs['cls_boxes'][object_ind][:4] / scale
                for i in [0, 2]:
                    bbox[i] = np.maximum(np.minimum(bbox[i], img_shape[1]-1), 0)
                    bbox[i + 1] = np.maximum(
                        np.minimum(bbox[i + 1], img_shape[0]-1),
                        0
                    )
                bbox_ltwh = xyxy_to_xywh(bbox)
                bbox_ltwh = np.round(bbox_ltwh).astype(np.int32)
                keypoints = np.array(
                    [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]], dtype=np.float32)
           
                entry['score'] = score
                entry['bbox_ltwh'] = bbox_ltwh
                entry['keypoints'] = keypoints
                entry['keypoints_score'] = np.array([[score]], dtype=np.float32)
                people.append(entry)
                index += 1

    return {'people': people}


