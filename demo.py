# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import torchvision.datasets as dset
#from scipy.misc import imread
from imageio import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="/srv/share/jyang375/models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


def check_overlap(box1, box2):
  check_x = box1[2] >= box2[0] and box2[2] >= box1[0]
  check_y = box1[3] >= box2[1] and box2[3] >= box1[1]
  if check_x and check_y:
    return True
  else:
    return False
def iou(bb1, bb2):#########################
  """ check if overlap"""
  #assert bb1[0] < bb1[2]
  #assert bb1[1] < bb1[3]
  #assert bb2[0] < bb2[2]
  #assert bb2[1] < bb2[3]
  # determine the coordinates of the intersection rectangle
  #print(bb1, bb2)
  
  x_left = max(bb1[0], bb2[0])
  y_top = max(bb1[1], bb2[1])
  x_right = min(bb1[2], bb2[2])
  y_bottom = min(bb1[3], bb2[3])
  iw = float(x_right - x_left)
  ih = float(y_bottom - y_top) 
  inters = iw * ih
  #print(type(bb1[0]), type(bb2[0]))
  # union
  uni = (float(bb1[2]-bb1[0])*float(bb1[3]-bb1[1]) + float(bb2[2]-bb2[0])*float(bb2[3]-bb2[1]) - inters)
  overlaps = inters / uni
  #print(overlaps)
  return overlaps

def MaximumBox(box1, box2):
  x1 = min(box1[0],box2[0])
  y1 = min(box1[1],box2[1])
  x2 = max(box1[2],box2[2])
  y2 = max(box1[3],box2[3])
  return [x1,y1,x2,y2]

def cal_accuracy(data, img_name, mode = "all", partial_flag = False):
  # data = {"class1": [[x1,y1,x2,y2,score],[],...], "class2": [[]..]..} (for all)
  # data = {"class1": [x1,y1,x2,y2,score], "class2": [], ..} (for full)
  acc = 0
  TP_FP_FN = 0
  count_all_pdt = 0
  counter = 0
  path_of_xml = "/home/jason/Faster-R-CNN/data/VOCdevkit2007/VOC2007/Annotations/"
  filename = os.path.join(path_of_xml, img_name + '.xml')
  tree = ET.parse(filename)
  objs = tree.findall('object')
  num_objs = len(objs)
  boxes = np.zeros((num_objs, 4), dtype=np.uint16)
  gt_classes = [None]*(num_objs)
  for ix, obj in enumerate(objs):
    bbox = obj.find('bndbox')
    # Make pixel indexes 0-based
    x1 = float(bbox.find('xmin').text) - 1
    y1 = float(bbox.find('ymin').text) - 1
    x2 = float(bbox.find('xmax').text) - 1
    y2 = float(bbox.find('ymax').text) - 1
    
    cls = obj.find('name').text.strip()
    boxes[ix, :] = [x1, y1, x2, y2]
    gt_classes[ix] = cls
    if gt_classes[ix][:4] == "H.sp":
      n = 4
    else:
      n = 5
    #print(cls)
    #print(boxes[ix, :])
    if gt_classes[ix][:n] in data.keys() and mode == "full":#mode full means only one label would be count
      #print(data[gt_classes[ix]], boxes[ix, :])
      if iou(data[gt_classes[ix][:n]], boxes[ix, :]) > 0.2 \
         and check_overlap(data[gt_classes[ix][:n]], boxes[ix, :])\
         and not partial_flag :## if full-length label, all labels true
        acc = 3
        counter = 3
        break
      elif iou(data[gt_classes[ix][:n]], boxes[ix, :]) > 0.2 \
         and check_overlap(data[gt_classes[ix][:n]], boxes[ix, :])\
         and partial_flag :## if partial lable, only one label true
        acc = 1
        counter = 1
        break
      elif not partial_flag:
        counter+=1
      else:
        counter = 1
        break
    elif gt_classes[ix] in data.keys() and mode == "all":
      for i in range(len(data[gt_classes[ix]])):
        if iou(data[gt_classes[ix]][i], boxes[ix, :]) > 0.2 \
           and check_overlap(data[gt_classes[ix]][i], boxes[ix, :]):
          acc += 1
    else:
      counter+=1
      continue
  TP_FP_FN+=counter 
  data_values = list(data.values())
  for i in range(len(data_values)):
    count_all_pdt = count_all_pdt + len(data_values[i])
  if mode == "full":
    TP = acc
    return TP, TP_FP_FN
  elif mode == "all":
    TP = acc
    TP_FP_FN = count_all_pdt + ix+1 - acc
    
    return TP, TP_FP_FN
  else:
    acc = None
    return None, None


    
if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  pascal_classes = np.asarray(['__background__',  # always index 0
                         'A.bes(H)','A.bes(T)','A.bes','A.bic(H)','A.bic(T)','A.bic',
                         'A.fuj(H)','A.fuj(T)','A.fuj','B.xyl(H)','B.xyl(T)','B.xyl',
                         'C.ele(H)','C.ele(T)','C.ele','M.ent(H)','M.ent(T)','M.ent',
                         'M.gra(H)','M.gra(T)','M.gra','M.inc(H)','M.inc(T)','M.inc',
                         'P.cof(H)','P.cof(T)','P.cof','P.vul(H)','P.vul(T)','P.vul',
                         'P.spe(H)','P.spe(T)','P.spe','H.sp(H)','H.sp(T)','H.sp',
                         'M.ams(H)' ,'M.ams(T)','M.ams'])###################

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  if args.cuda > 0:
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  # pdb.set_trace()

  print("load checkpoint %s" % (load_name))

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  with torch.no_grad():
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()
  max_per_image = 100
  thresh = 0.8
  vis = True

  webcam_num = args.webcam_num
  # Set up webcam or get image directories
  if webcam_num >= 0 :
    cap = cv2.VideoCapture(webcam_num)
    num_images = 0
  else:
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)

  print('Loaded Photo: {} images.'.format(num_images))

  c = 0
  #accs1 = 0
  #accs2 = 0
  TP1s = 0
  TP2s = 0
  TP_FP_FN1s = 0
  TP_FP_FN2s = 0
  while (num_images > 0):
      total_tic = time.time()
      if webcam_num == -1:
        num_images -= 1

      # Get image from the webcam
      if webcam_num >= 0:
        if not cap.isOpened():
          raise RuntimeError("Webcam could not open. Please check connection.")
        ret, frame = cap.read()
        im_in = np.array(frame)
      # Load the demo image
      else:
        im_file = os.path.join(args.image_dir, imglist[num_images])
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))
      if len(im_in.shape) == 2:
        im_in = im_in[:,:,np.newaxis]
        im_in = np.concatenate((im_in,im_in,im_in), axis=2)
      # rgb -> bgr
      im = im_in[:,:,::-1]

      blobs, im_scales = _get_image_blob(im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)

      with torch.no_grad():
          im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
          im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
          gt_boxes.resize_(1, 1, 5).zero_()
          num_boxes.resize_(1).zero_()
      #print(im_blob[0].shape)
      cv2.imwrite('test.jpg',im_blob[0]) 

      # pdb.set_trace()
      det_tic = time.time()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= im_scales[0]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      
      
      new_pred_boxes = torch.cuda.FloatTensor(300, 160).zero_()##############################
      new_scores = torch.cuda.FloatTensor(300,40).zero_()
      for k in range(13):
        b = torch.cat((pred_boxes[:,12*k+4:12*k+8],pred_boxes[:,12*k+8:12*k+12]),0)
        s = torch.cat((scores[:,3*k+1],scores[:,3*k+2]),0)
        keep = nms(b, s, 0.2)
        #new head class
        idx = [g for g in range(len(keep)) if keep[g] <300]
        new_pred_boxes[:len(keep[idx]),12*k+4:12*k+8] = b[keep[idx]]
        new_scores[:len(keep[idx]),3*k+1] = s[keep[idx]]
        #new tail class
        idx = [g for g in range(len(keep)) if keep[g] >=300]
        new_pred_boxes[:len(keep[idx]),12*k+8:12*k+12] = b[keep[idx]]
        new_scores[:len(keep[idx]),3*k+2] = s[keep[idx]]
        #new full length class = original
        new_pred_boxes[:,12*k+12:12*k+16] = pred_boxes[:,12*k+12:12*k+16]
        new_scores[:,3*k+3] = scores[:,3*k+3]
      '''
      new_pred_boxes = torch.cuda.FloatTensor(300, 160).zero_()############################## nms for all head, (tail, full-length) classes
      new_scores = torch.cuda.FloatTensor(300,40).zero_()
      for j in range(3):
        b = pred_boxes[:,4*j+4:4*j+8]
        s = scores[:,j+1]
        #print(b.shape)
        for k in range(1, 13):
          b = torch.cat((b, pred_boxes[:,12*k+4*j+4:12*k+4*j+8]),0)
          s = torch.cat((s ,scores[:,3*k+j+1]),0)
        #print(b.shape,s.shape)
        #sys.exit()
        keep = nms(b, s, 0.5)
        #print(keep, len(keep))
        for l in range(13):
          idx = [g.item() for g in keep if g < (l+1)*300 and g >= l*300]
          #print(len(idx), new_pred_boxes[:len(idx),12*l+4*j+4:12*l+4*j+8])
          #print(new_pred_boxes[:len(idx),12*l+4*j+4:12*l+4*j+8].shape, b[idx].shape)
          new_pred_boxes[:len(idx),12*l+4*j+4:12*l+4*j+8] = b[idx]
          new_scores[:len(idx),3*l+j+1] = s[idx]
          #print([g for g in s[idx] if g > 0.5])
      '''
      #new_pred_boxes = pred_boxes
      #new_scores = scores
      new_pred_boxes = new_pred_boxes.cpu()
      new_scores = new_scores.cpu()
      voting_data = {}     
      J = 0
      partial_flag = False
      if vis:
          im2show = np.copy(im)
      for j in xrange(1, len(pascal_classes)):
          inds = torch.nonzero(new_scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = new_scores[:,j][inds]          
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = new_pred_boxes[inds, :]
            else:
              cls_boxes = new_pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            voting_data[pascal_classes[j]] = cls_dets
            #print(cls_dets.shape)
            if vis:
              img_name = imglist[num_images][:-4]
              im2show = cv2.UMat(im2show).get()
              #print(type(im2show))  
              im2show, _ , J= vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(),img_name,  0.8, J)
              J+=1
              #f = open("species.txt", "w+")
              #f.write(pascal_classes[j])
      # voting_data['class']: [[a1,y1,x2,y2,score],[],..]
      #TP1, TP_FP_FN1 = cal_accuracy(voting_data, imglist[num_images][:-4], "all", partial_flag)
      #TP1s +=TP1 
      #TP_FP_FN1s += TP_FP_FN1
      # initial voting_results
      #print("vdata: "+ str(voting_data))
      voting_results = {}
      list_of_keys = list(voting_data.keys())
      list_of_values = list(voting_data.values())
      # initial flag 
      flag = [None]*len(list_of_values)
      for m in range(len(list_of_values)):
        flag[m] = [0]*len(list_of_values[m])
      group_count = 0
      species_List = []
      species_List.extend([x for x in range(0,len(list_of_values)) if list_of_keys[x][-1]!=")"])
      #print(species_List)
      species_List.extend([x for x in range(0,len(list_of_values)) if list_of_keys[x][-1]==")"])
      #print(species_List)
      
      
      ############################   decide how many nematodes and allocate boxes
      for m in species_List:
        for l in range(len(list_of_values[m])):         
          for n in species_List:
            for k in range(len(list_of_values[n])):
              if m != n or l != k:
                if group_count == 0: # decide the first group                             
                  if check_overlap(list_of_values[m][l][0:4], list_of_values[n][k][0:4]) == True:  
                    max_box = MaximumBox(list_of_values[m][l], list_of_values[n][k])
                    voting_results["group"+str(group_count)] = {"region": max_box, \
                                                                list_of_keys[m]: list_of_values[m][l][4],\
                                                                list_of_keys[n]: list_of_values[n][k][4]}
                    flag[m][l] = 1+group_count
                    flag[n][k] = 1+group_count
                    group_count+=1
                  elif m == n:# same class but not overlap
                    if list_of_keys[m][-1] != ")":# full-length case => two nematodes
                      reg = [x for x in list_of_values[m][l][0:4]]
                      voting_results["group"+str(group_count)] = {"region": reg, \
                                                                  list_of_keys[m]: list_of_values[m][l][4]}
                      flag[m][l] = 1+group_count
                      group_count+=1
                      reg = [x for x in list_of_values[n][k][0:4]]
                      voting_results["group"+str(group_count)] = {"region": reg, \
                                                                  list_of_keys[n]: list_of_values[n][k][4]}
                      flag[n][k] = 1+group_count
                      group_count+=1
                    elif list_of_values[m][l][4] >= list_of_values[n][k][4]: # partial case => higher score
                      reg = [x for x in list_of_values[m][l][0:4]]
                      voting_results["group"+str(group_count)] = {"region": reg, \
                                                                  list_of_keys[m]: list_of_values[m][l][4]}
                      flag[m][l] = 1+group_count
                      group_count+=1
                    else:
                      reg = [x for x in list_of_values[n][k][0:4]]
                      voting_results["group"+str(group_count)] = {"region": reg, \
                                                                  list_of_keys[n]: list_of_values[n][k][4]}
                      flag[n][k] = 1+group_count 
                      group_count+=1
                  else:# different classes and not overlap => first box create a group
                    reg = [x for x in list_of_values[m][l][0:4]]
                    voting_results["group"+str(group_count)] = {"region": reg,\
                                                                list_of_keys[m]:list_of_values[m][l][4]}
                    flag[m][l] = 1+group_count                
                    group_count+=1
                else: # exist 1 or more groups
                  if flag[m][l] == 0:
                    for p in range(group_count):#find if overlap with previous groups
                      if check_overlap(list_of_values[m][l][0:4], voting_results["group"+str(p)]\
                                                                                ["region"]) == True:
                        max_box = MaximumBox(list_of_values[m][l], voting_results["group"+str(p)]["region"])
                        voting_results["group"+str(p)]["region"] = max_box
                        if list_of_keys[m] in voting_results["group"+str(p)].keys():                        
                          voting_results["group"+str(p)][list_of_keys[m]] = max(list_of_values[m][l][4], \
                                                        voting_results["group"+str(p)][list_of_keys[m]])
                          flag[m][l] = 1+p
                          break
                        else:                                                
                          voting_results["group"+str(p)][list_of_keys[m]] = list_of_values[m][l][4]
                          flag[m][l] = 1+p
                          break
                    # not overlap with previous groups & full-length=> create a new group
                    if flag[m][l] == 0 and list_of_keys[m][-1] != ")":
                      reg = [x for x in list_of_values[m][l][0:4]]
                      voting_results["group"+str(group_count)] = {"region": reg,\
                                                                  list_of_keys[m]:list_of_values[m][l][4]}
                      flag[m][l] = 1+group_count
                      group_count+=1
                    elif flag[m][l] == 0 and list_of_keys[m][-1] == ")":
                      if flag[n][k] == 0 and check_overlap(list_of_values[m][l][0:4],\
                         list_of_values[n][k][0:4]) == True and (m != n or l != k):
                        max_box = MaximumBox(list_of_values[m][l], list_of_values[n][k])
                        voting_results["group"+str(group_count)] = {"region": max_box, \
                                                                    list_of_keys[m]: list_of_values[m][l][4],\
                                                                    list_of_keys[n]: list_of_values[n][k][4]}
                        flag[m][l] = 1+group_count
                        flag[n][k] = 1+group_count
                        group_count+=1
                    else:
                      continue
                  else:
                    continue
      if len(species_List) == 1 and len(list_of_values[0]) ==1:
        voting_results["group0"] = {"region": [s for s in list_of_values[0][0][0:4]], \
                                    list_of_keys[0]: list_of_values[0][0][4]}
        group_count = 1
      '''              
      #print(imglist[num_images][:-4]+': '+str(voting_results)+"\n")       
      if group_count == 1:# delete the case of single-partial group
        zero_flag = 1
        v_list = list(voting_results.values())
        v_list = v_list[0]
        v_list = list(v_list.keys())
        #print(v_list)
        for m in range(1, len(v_list)):
          if v_list[m][-1] != ')':
            zero_flag = 0
            break
        if zero_flag == 1:
          voting_results = {}
          group_count = 0
      '''
      # calculate the score for each nematode  
      groups = list(voting_results.keys())
      datas = list(voting_results.values()) 
      #print("gresult: "+str(voting_results))
      results = {}
      if group_count != 0:
        c+=len(groups)
        for k in range(len(groups)):
          scores = list(datas[k].values())#score[0]: region of group[x1, y1, x2, y2]
          species = list(datas[k].keys())
          #print(scores[0])
          for m in range(1, len(species)): 
            if species[m][:4] != "H.sp":
              s = species[m][:5]
              results[s] = copy.deepcopy(scores[0])
              #print(results[s])
              if len(results[s]) == 4:
                results[s].extend([0])
            else:
              s = species[m][:4]
              results[s] = scores[0]
              if len(results[s]) == 4:
                results[s].extend([0])
          # results = [x1,y1,x2,y2,score]  
          for m in range(1, len(species)):
            if species[m][:4] != "H.sp":
              s = species[m][:5]
            else:
              s = species[m][:4]
            if len(species) == 2 and species[m][-1] == ")":
              results[s][4] = scores[m]
              partial_flag = True
              break
            elif species[m][-2] == "H":
              results[s][4] = results[s][4] + scores[m]*0.2
            elif species[m][-2] == "T":
              results[s][4] = results[s][4] + scores[m]*0.2
            else:
              results[s][4] = results[s][4] + scores[m]*0.6
          
          #print("vresult: "+str(results))
                              
          list_of_species = list(results.keys())
          list_of_scores = list(results.values())
          #print(list_of_scores) 
          highest_data = {}
          highest_score = [0,0,0,0,0]
          highest_class = ""
          for m in range(len(list_of_species)):# find the highest score class 
            #print(list_of_scores[m], highest_score)           
            if list_of_scores[m][4]>highest_score[4]:
              
              if im_in.shape == (4000,6000,3):
                highest_score = list_of_scores[m]
              else:
                highest_score = [list_of_scores[m][1],4000-list_of_scores[m][2],list_of_scores[m][3],4000-list_of_scores[m][0], list_of_scores[m][4]]
              
              #highest_score = list_of_scores[m]
              highest_class = list_of_species[m]
              
          highest_data[highest_class] = highest_score
          #print(imglist[num_images][:-4])
          #print("highest: "+str(highest_data))
          TP2, TP_FP_FN2 = cal_accuracy(highest_data, imglist[num_images][:-4], "full", partial_flag)
          TP2s += TP2 
          TP_FP_FN2s += TP_FP_FN2
          #print(TP2s, TP_FP_FN2s)
          height = int(im_info_np[0][0]/im_info_np[0][2])
          #print(height)
          cv2.putText(im2show, "%d" % k, (int(scores[0][0]),int(scores[0][1]- 80)), \
                                          cv2.FONT_HERSHEY_PLAIN, 3, (251,9,3), thickness=3)
          
          cv2.putText(im2show, "%d" % k, (10+k*300, height-200), \
                      cv2.FONT_HERSHEY_PLAIN, 3, (251,9,3), thickness=3)
          for m in range(len(list_of_species)):
            #print(list_of_scores[m])          
            cv2.putText(im2show, '%s: %.2f' % (list_of_species[m], list_of_scores[m][4]), \
                        (10+k*300, height-160+m*40), \
                         cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), thickness=3)
          
          results = {}
      misc_toc = time.time()
      nms_time = misc_toc - misc_tic
      
      if webcam_num == -1:
          sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                           .format(num_images + 1, len(imglist), detect_time, nms_time))
          sys.stdout.flush()
      
      if vis and webcam_num == -1:
          # cv2.imshow('test', im2show)
          # cv2.waitKey(0)
          result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
          im2show = np.asarray(im2show)
          #cv2.imwrite(result_path, im2show)
      else:
          im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
          cv2.imshow("frame", im2showRGB)
          total_toc = time.time()
          total_time = total_toc - total_tic
          frame_rate = 1 / total_time
          print('Frame rate:', frame_rate)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
  #print(accs1, len(imglist))
  #print(accs2, c)
  #print("Accuracy before voting: %.3f" % (TP1s/TP_FP_FN1s))
  print("\n Accuracy after voting: %.3f"% (TP2s/TP_FP_FN2s))
  if webcam_num >= 0:
      cap.release()
      cv2.destroyAllWindows()
