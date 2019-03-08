import os
import json
import argparse
#from time import time
import time
import cv2
import torch
from torch import nn
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from visualization import visualize_annotations, add_label
from lib.utils.config_parse import cfg_from_file
from lib.ssds_train import test_model
from lib.utils.data_augment import preproc
from lib.utils.config_parse import cfg
from lib.modeling.model_builder import create_model
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from postprocessing import postprocess
from lib.utils.eval_utils import *
from lib.utils.visualize_utils import *
from lib.layers import *



def resume_checkpoint(model, resume_checkpoint):
    if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
        print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
        return False
    print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
    checkpoint = torch.load(resume_checkpoint)

    # print("=> Weigths in the checkpoints:")
    # print([k for k, v in list(checkpoint.items())])

    # remove the module in the parrallel model
    if 'module.' in list(checkpoint.items())[0][0]:
        pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        checkpoint = pretrained_dict

    # change the name of the weights which exists in other model
    # change_dict = {
    #         'conv1.weight':'base.0.weight',
    #         'bn1.running_mean':'base.1.running_mean',
    #         'bn1.running_var':'base.1.running_var',
    #         'bn1.bias':'base.1.bias',
    #         'bn1.weight':'base.1.weight',
    #         }
    # for k, v in list(checkpoint.items()):
    #     for _k, _v in list(change_dict.items()):
    #         if _k == k:
    #             new_key = k.replace(_k, _v)
    #             checkpoint[new_key] = checkpoint.pop(k)
    # change_dict = {'layer1.{:d}.'.format(i):'base.{:d}.'.format(i+4) for i in range(20)}
    # change_dict.update({'layer2.{:d}.'.format(i):'base.{:d}.'.format(i+7) for i in range(20)})
    # change_dict.update({'layer3.{:d}.'.format(i):'base.{:d}.'.format(i+11) for i in range(30)})
    # for k, v in list(checkpoint.items()):
    #     for _k, _v in list(change_dict.items()):
    #         if _k in k:
    #             new_key = k.replace(_k, _v)
    #             checkpoint[new_key] = checkpoint.pop(k)

    resume_scope = cfg.TRAIN.RESUME_SCOPE
    # extract the weights based on the resume scope
    if resume_scope != '':
        pretrained_dict = {}
        for k, v in list(checkpoint.items()):
            for resume_key in resume_scope.split(','):
                if resume_key in k:
                    pretrained_dict[k] = v
                    break
        checkpoint = pretrained_dict

    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
    # print("=> Resume weigths:")
    # print([k for k, v in list(pretrained_dict.items())])

    checkpoint = model.state_dict()

    unresume_dict = set(checkpoint)-set(pretrained_dict)
    if len(unresume_dict) != 0:
        print("=> UNResume weigths:")
        print(unresume_dict)

    checkpoint.update(pretrained_dict)

    return model.load_state_dict(checkpoint)
    
# def test_epoch(model, data_loader, detector, output_dir, use_gpu):
#     model.eval()

#     dataset = data_loader.dataset
#     num_images = len(dataset)
#     num_classes = detector.num_classes
#     all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
#     empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))

#     _t = Timer()
#     print(num_images)
#     for i in iter(range((num_images))):
#         img = dataset.pull_image(i)
#         scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
#         if use_gpu:
#             images = Variable(dataset.preproc(img)[0].unsqueeze(0).cuda(), volatile=True)
#         else:
#             images = Variable(dataset.preproc(img)[0].unsqueeze(0), volatile=True)

#         _t.tic()
#         # forward
#         out = model(images, phase='eval')

#         # detect
#         detections = detector.forward(out)

#         time = _t.toc()

#         # TODO: make it smart:
#         for j in range(1, num_classes):
#             cls_dets = list()
#             for det in detections[0][j]:
#                 if det[0] > 0:
#                     d = det.cpu().numpy()
#                     score, box = d[0], d[1:]
#                     box *= scale
#                     box = np.append(box, score)
#                     cls_dets.append(box)
#             if len(cls_dets) == 0:
#                 cls_dets = empty_array
#             all_boxes[j][i] = np.array(cls_dets)

#         # log per iter
#         log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]\r'.format(
#                 prograss='#'*int(round(10*i/num_images)) + '-'*int(round(10*(1-i/num_images))), iters=i, epoch_size=num_images,
#                 time=time)
#         sys.stdout.write(log)
#         sys.stdout.flush()

#     # write result to pkl
#     with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
#         pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

#     # currently the COCO dataset do not return the mean ap or ap 0.5:0.95 values
#     print('Evaluating detections')
#     data_loader.dataset.evaluate_detections(all_boxes, output_dir)
    
class Labtesteval(COCOeval):
    """Evaluation tool slightly modified from original pycocotools.cocoeval"""
    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox'):
        super(Labtesteval, self).__init__(cocoGt, cocoDt, iouType)

    def record(self, iouThr=0.75, areaRng='all', maxDets=100):
        '''record all failure cases'''
        record_list = []
        p = self.params
        numImgs = len(p.imgIds)
        areaInd = p.areaRngLbl.index(areaRng)
        iouInd = np.where(p.iouThrs==iouThr)[0][0]
        for evalImg in self.evalImgs[areaInd*numImgs:(areaInd+1)*numImgs]:
            # extract eval info
            dtScores = evalImg['dtScores']
            inds = np.argsort(dtScores, kind='mergesort')[:maxDets]
            dtm = evalImg['dtMatches'][iouInd, inds]
            dtIg = evalImg['dtIgnore'][iouInd, inds]
            gtm = evalImg['gtMatches'][iouInd]
            gtIg = evalImg['gtIgnore']

            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
            fns = np.logical_and(np.logical_not(gtm), np.logical_not(gtIg))

            if np.any(fps) or np.any(fns):
                record_list.append({
                    'image_id': int(evalImg['image_id']),
                    'gt_anns': self.cocoGt.loadAnns(ids=evalImg['gtIds']),
                    'dt_anns': self.cocoDt.loadAnns(ids=evalImg['dtIds'])
                    })
        return record_list

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            # modified this equation for labtest footage
            stats = np.zeros((4,))
            stats[0] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--data_dir', dest='data_dir', default = '/data/tracking_data/meeting_room/tracking_data_meeting_room_easy', help='name of config file',
                        type=str)
    parser.add_argument('--gpu', dest='gpu', help='gpu index assigned to model',
                        type=str, default='3')
    parser.add_argument('--cfg', dest='config_file',
            help='optional config file', default=None, type=str) 
    parser.add_argument('--weight', dest='weight',
            help='weight file', default='./experiments/models/darknet_53_yolo_v3_aifi/yolo_v3_darknet_53_coco_epoch_116.pth', type=str)                                    
    parser.add_argument('--replace', dest='replace', help='set True to ignore any existing results and run detection model',
                        action='store_true', default=False)
    parser.add_argument('--mode', dest='mode', help='choose between "eval" or "replay"',
                        type=str, default='eval')
    parser.add_argument('--imgs', dest='num_imgs', help='number of images loaded for eval. use -1 to include all images',
                        type=int, default=-1)
    parser.add_argument('--result_name', dest='result_name', default = 'name', help='name of detection result',
                        type=str)
    args = parser.parse_args()
    return args


def run_evaluation(coco_gt, json_dt, json_fails, iouType, catIds, imgIds):
    # run evaluation
    coco_dt = coco_gt.loadRes(json_dt)
    coco_eval = Labtesteval(cocoGt=coco_gt, cocoDt=coco_dt, iouType=iouType)

    coco_eval.params.catIds = catIds
    coco_eval.params.imgIds = imgIds

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    failure_list = coco_eval.record()
    with open(json_fails, 'w') as f:
        json.dump(failure_list, f)
    print('failure cases stored in {}'.format(json_fails))


def run_detection(data_dir, coco_gt, im_ids):
 
    model, priorbox = create_model(cfg.MODEL)
    priors = Variable(priorbox.forward(), volatile=True)
    detector = Detect(cfg.POST_PROCESS, priors)

    # Utilize GPUs for computation
    model.cuda()
    priors.cuda()
    cudnn.benchmark = True     
    preprocess = preproc(cfg.DATASET.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS, -2)
                       
    resume_checkpoint(model, args.weight)   
                        
    num_classes = 2                   
    results = []
    time_all = []
    time_per_step={"nms_time":[],"cpu_tims":[], "scores_time":[],"box_time":[],"gpunms_time":[],
    "base_time":[],"extra_time":[],"head_time":[]}
    for i, index in enumerate(im_ids):
        # load img
        print('evaluating image {}/{}'.format(i, len(im_ids)))
        im_data = coco_gt.loadImgs(ids=index)[0]
        img = cv2.imread(os.path.join(data_dir, 'frames', im_data['file_name']))
        scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        img_shape = img.shape
        images = Variable(preprocess(img)[0].unsqueeze(0).cuda(), volatile=True)
        img_dict = {
            'version': 0,
            'time': 0.,
            'camera_id': 0,
            'image': img
        }
        # run detection model
        torch.cuda.synchronize()
        time_all_start = time.perf_counter()

        # forward
        out,base_time,extra_time,head_time = model(images, phase='eval')

        # detect
        detections,nms_time, cpu_tims, scores_time,box_time,gpunms_time = detector.forward(out)

        torch.cuda.synchronize()
        time_all_end = time.perf_counter()

        time_all.append(1000 * (time_all_end - time_all_start))
        time_per_step["nms_time"].append(1000 *nms_time)
        time_per_step["cpu_tims"].append(1000 *cpu_tims)
        time_per_step["scores_time"].append(1000 *scores_time)
        time_per_step["box_time"].append(1000 *box_time)
        time_per_step["gpunms_time"].append(1000 *gpunms_time)
        time_per_step["base_time"].append(base_time)
        time_per_step["extra_time"].append(extra_time)
        time_per_step["head_time"].append(head_time)

        scores = []
        cls_boxes = []
        for det in detections[0][1]:
            if det[0] > 0:
                d = det.cpu().numpy()
                score, box = d[0], d[1:]
                box *= scale
                scores.append(score)
                cls_boxes.append(box)
                #print(score)
                #print(box)
                
        output_blobs = {}                   
        output_blobs['scores'] = scores
        output_blobs['cls_boxes'] = cls_boxes
        print(np.array(cls_boxes).shape)
        output_dict = postprocess(output_blobs, 1., img_shape)
                        
        if len(output_dict['people']) == 0:
            continue
        # save result
        entry_index = 0
        for person in output_dict['people']:
            entry_result = {
                "image_id": index,
                "category_id": 1,
                "bbox": person['bbox_ltwh'].tolist(),
                "score": person['score']}
            results.append(entry_result)
    # save results as json file
    with open(json_dt, 'w') as f:
        json.dump(results, f)
    print('detection results saved in {}'.format(json_dt))
    print('average running time: {}ms'.format(sum(time_all) / len(time_all))) 
    for key in time_per_step.keys():
        print(key,' average running time: {}ms'.format(sum(time_per_step[key]) / len(time_per_step[key])))
        #print('average running time: {}ms'.format(sum(time_per_step[key]) / len(time_per_step[key])

if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_imgs = args.num_imgs
    mode = args.mode
    cfg_from_file(args.config_file)

    data_dir = args.data_dir
    data_type = data_dir.split('/')[-1]

    # dir of gt jsonfile and dt jsonfile
    json_gt = os.path.join(data_dir, 'annotations', 'annotations.json')
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    json_dt = 'outputs/{}_{}results.json'.format(data_type,args.result_name)
    json_fails = 'outputs/{}_{}failures.json'.format(data_type,args.result_name)

    # load gt file
    coco_gt = COCO(json_gt)
    catIds = coco_gt.getCatIds('head')
    im_ids = coco_gt.getImgIds(catIds=catIds)[
        :num_imgs] if num_imgs != -1 else coco_gt.getImgIds(catIds=catIds)

    if mode == 'eval':
        print('Start evaluating: {}'.format(data_type))
        # check if results file already exists
        if os.path.isfile(json_dt) and not args.replace:
            print("result file already exists. Will skip detection...")
        else:
            run_detection(data_dir, coco_gt, im_ids)
        # run evaluation
        run_evaluation(coco_gt, json_dt, json_fails, "bbox", catIds, im_ids)
        print('evaluation completed.')
    elif mode == 'replay':
        if not os.path.isdir('outputs/fails'):
            os.mkdir('outputs/fails')
        print('Start replaying failure cases from {}'.format(json_fails))
        with open(json_fails, 'r') as f:
            failed_data = json.load(f)
            failed_data.sort(key=lambda x: x['image_id'])
            for entry in failed_data:
                image_name = coco_gt.loadImgs(ids=entry['image_id'])[0]['file_name']
                img = cv2.imread(os.path.join(data_dir, 'frames', image_name))
                assert img is not None
                add_label(img, image_name, (0, 20), (0, 255, 0), 2)
                img = visualize_annotations(img, entry['gt_anns'], (0, 255, 0), alpha=1.0)
                img = visualize_annotations(img, entry['dt_anns'], (0, 0, 255), alpha=0.5)
                cv2.imshow(json_fails, img)
                key = cv2.waitKey(-1)
                cv2.imwrite('outputs/fails/{}'.format(image_name.replace('/', '_')), img)
                if key == ord('q'):
                    break
        cv2.destroyAllWindows()

