# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from PIL import Image
from torchvision.utils import save_image
import cv2
from torch.utils.data import DataLoader
from srgan_datasets import *
from srgan import *
import torch.nn.functional as F
from datasets.voc_eval import parse_rec
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

classes = ('__background__',  # always index 0
           'A.bes(H)','A.bes(T)','A.bes','A.bic(H)','A.bic(T)','A.bic',
           'A.fuj(H)','A.fuj(T)','A.fuj','B.xyl(H)','B.xyl(T)','B.xyl',
           'C.ele(H)','C.ele(T)','C.ele','M.ent(H)','M.ent(T)','M.ent',
           'M.gra(H)','M.gra(T)','M.gra','M.inc(H)','M.inc(T)','M.inc',
           'P.cof(H)','P.cof(T)','P.cof','P.vul(H)','P.vul(T)','P.vul',
           'P.spe(H)','P.spe(T)','P.spe','H.sp(H)','H.sp(T)','H.sp',
           'M.ams(H)' ,'M.ams(T)','M.ams'                      
           )###################
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
                      default='cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
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
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args



def parse_args_for_srgan():
  os.makedirs("srgan/images", exist_ok=True)
  os.makedirs("srgan/saved_models", exist_ok=True)
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--epoch", type=int, default=500 , help="epoch to start training from")
  parser.add_argument("--n_epochs", type=int, default=501, help="number of epochs of training")
  parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
  parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
  parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
  parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
  parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
  parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
  #parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
  parser.add_argument("--hr_height", type=int, default=1024, help="high res. image height")
  parser.add_argument("--hr_width", type=int, default=1024, help="high res. image width")
  parser.add_argument("--channels", type=int, default=3, help="number of image channels")
  parser.add_argument("--sample_interval", type=int, default=50, help="interval between saving image samples")
  parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between model checkpoints")
  
  opt = parser.parse_args([])

  return opt
  
  
  
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def load_gt_box(annopath,
             imagesetfile,
             classname,
             cachedir):
  if not os.path.isdir(cachedir):
      os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')
  #print(recs)
  # extract gt objects for this class
  class_recs = {}
  npos = 0
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}
  #print(class_recs)
  #print(len(class_recs))
  return class_recs


def iou(bb1, bb2):#########################
  """ check if overlap"""
  #assert bb1[0] < bb1[2]
  #assert bb1[1] < bb1[3]
  #assert bb2[0] < bb2[2]
  #assert bb2[1] < bb2[3]

  # determine the coordinates of the intersection rectangle
  #print(bb1[0], bb2[0])
  x_left = max(bb1[0], bb2[0])
  y_top = max(bb1[1], bb2[1])
  x_right = min(bb1[2], bb2[2])
  y_bottom = min(bb1[3], bb2[3])
  iw = x_right - x_left
  ih = y_bottom - y_top  
  inters = iw * ih
  # union
  uni = ((bb1[2]-bb1[0])*(bb1[3]-bb1[1]) + (bb2[2]-bb2[0])*(bb2[3]-bb2[1]) - inters)

  overlaps = inters / uni
  return overlaps

def Area(vertex):
  width = vertex[2] - vertex[0]
  height = vertex[3] - vertex[1]
  area = width*height
  return area

if __name__ == '__main__':

  args = parse_args()
  args_sr = parse_args_for_srgan()
  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.0
  else:
    thresh = 0.0

  save_name = 'faster_rcnn_10'
  num_images = len(imdb.image_index)

  all_boxes = [[[] for _ in range(num_images)]
               for _ in range(imdb.num_classes)]
  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in range(num_images):

      data = next(data_iter)
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(data[2].size()).copy_(data[2])
              num_boxes.resize_(data[3].size()).copy_(data[3])
      
      #print(im_data.shape)
      #print(im_info.shape)
      #print(gt_boxes)
      #print(num_boxes)
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
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      
      #print(scores[:,1:3].shape)
      #print(pred_boxes[:,4:12].shape)
      ##############################   decline head-tail overlapping
      new_pred_boxes = torch.cuda.FloatTensor(300, 160).zero_()
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
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
      
      for j in range(1, imdb.num_classes):
          inds = torch.nonzero(new_scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = new_scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = new_pred_boxes[inds, :]
            else:
              cls_boxes = new_pred_boxes[inds][:, j * 4:(j + 1) * 4]
            #print(cls_boxes.shape)
            #print(cls_scores.unsqueeze(1).shape)
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            
            
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array
      #print(exist_classes)     
      #for k, j in enumerate(exist_classes):
      #  all_boxes[j][i] = exist_dets[k]
      #print(all_boxes)
      
      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          #print(all_boxes[3][i][:,-1])
          image_scores = np.hstack([all_boxes[j][i][:,-1]
                                    for j in range(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

      if vis:
          cv2.imwrite('result.png', im2show)
          pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)
  #print(all_boxes[1][0][0])
  print(torch.cuda.current_device())
  with torch.cuda.device(torch.cuda.current_device()): 
    torch.cuda.empty_cache()
  #################################### filter imgs need to do SRGAN-preprocessing
  annopath = '/home/jason/faster-rcnn.pytorch-1.0/data/VOCdevkit2007/VOC2007/Annotations/{:s}.xml'
  imagesetfile = '/home/jason/faster-rcnn.pytorch-1.0/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
  cachedir = '/home/jason/faster-rcnn.pytorch-1.0/data/VOCdevkit2007/annotations_cache'
  image_file = '/home/jason/faster-rcnn.pytorch-1.0/data/VOCdevkit2007/VOC2007/JPEGImages'
  f = open(imagesetfile)
  new_indexes = []
  img_ids = []
  new_gt_boxes = []
  for line in f:
    img_ids.append(line.splitlines())
  img_ids = np.squeeze(img_ids)
  for i in range(num_images):
    for j in range(1, imdb.num_classes):
      gt_boxes_1 = load_gt_box(annopath,imagesetfile,classes[j],cachedir)
      if not np.any(all_boxes[j][i]):
        continue
                    
      if len(gt_boxes_1[img_ids[i]]['bbox']) == 0:
        continue        
      else:# 1 GT box in single image for a single class
        gt_b = gt_boxes_1[img_ids[i]]['bbox']
        #print(gt_b)
        z = 0
        for m in range(len(all_boxes[j][i])):
          for n in range(len(gt_b)):
            det_b = [int(l) for l in all_boxes[j][i][m][:4]]
            #print(all_boxes[j][i][m][4], iou(det_b, gt_b[n]), imdb.image_index[j])
            if all_boxes[j][i][m][4] > 0.5 and all_boxes[j][i][m][4] < 0.8 \
                and iou(det_b, gt_b[n]) > 0.5 and classes[j][-1]==")":
              print("srgan beginning......")      
              new_indexes.append(img_ids[i]+"_"+classes[j]+"_"+str(z))
              print(len(new_indexes))#, all_boxes[j][i][m][4], iou(det_b, gt_b[n]))
              img_path = os.path.join(image_file, img_ids[i]+".JPG")
              img = Image.open(img_path)
              img = np.asarray(img)
              quaterx = int(img.shape[1]*1/4)
              quatery = int(img.shape[0]*1/4)
              x1_padding = 0
              y1_padding = 0
              x2_padding = 0
              y2_padding = 0
              print(img.shape)
              if Area(det_b) >= Area(gt_b[n]):            
                x1, y1, x2, y2 = det_b
                print("det_b: " + str(det_b))
                if x1 > quaterx:
                  x1-=quaterx
                  x1_padding = quaterx
                else:
                  x1 = 0
                  x1_padding = x1
                if x2 < img.shape[0]-quaterx:
                  x2+= quaterx
                  x2_padding = quaterx
                else:
                  x2 = img.shape[0]-1
                  x2_padding = img.shape[0] - x2-1
                if y1 > quatery:
                  y1 -=quatery
                  y1_padding = quatery
                else:
                  y1 = 0
                  y1_padding = y1
                if y2 < img.shape[1]-quatery:
                  y2+=quatery
                  y2_padding = quatery
                else:
                  y2= img.shape[1]-1
                  y2_padding = img.shape[1] - y2-1
              else:
                x1, y1, x2, y2 = gt_b[n]
                print("gt_b: "+str(gt_b))
                if x1 > quaterx:
                  x1-=quaterx
                  x1_padding = quaterx
                else:
                  x1 = 0
                  x1_padding = x1
                if x2 < img.shape[0]-quaterx:
                  x2+= quaterx
                  x2_padding = quaterx
                else:
                  x2 = img.shape[0]-1
                  x2_padding = img.shape[0] - x2-1
                if y1 > quatery:
                  y1 -=quatery
                  y1_padding = quatery
                else:
                  y1 = 0
                  y1_padding = y1
                if y2 < img.shape[1]-quatery:
                  y2+=quatery
                  y2_padding = quatery
                else:
                  y2= img.shape[1]-1
                  y2_padding = img.shape[1] - y2-1
              x1, y1, x2, y2= int(x1),int(y1),int(x2), int(y2)  
                          
              new_gt_boxes.append([x1_padding, y1_padding, x2-x1-x1_padding-x2_padding, \
                                   y2-y1-y1_padding-y2_padding])# whole photo
              srgan_in = img[y1:y2 ,x1:x2 ,:]
              srgan_in = srgan_in[...,::-1]#rgb->bgr
              print(x1,y1,x2,y2,srgan_in.shape)
              cv2.imwrite(os.path.join("srgan/srgan_input", img_ids[i]+"_"+classes[j]+"_"+str(z)+".JPG"), srgan_in)
              print("save input: %s" %(img_ids[i]+"_"+classes[j]+"_"+str(z)))
              z+=1
              all_boxes[j][i][m] = np.append(gt_b[n], 1.0)# turn original pred box to gt box 
      
  with torch.cuda.device(torch.cuda.current_device()): 
    torch.cuda.empty_cache()       
  dataloader = DataLoader(
    ImageDataset("srgan/srgan_input", hr_shape=(1024,1024)),
    batch_size=1,
    shuffle=True,
    num_workers=0,
  )
  #gan_output = srgan(args_sr, dataloader)
  srgan(args_sr, dataloader)
  #print("length of data: %d"%len(gan_output))             
  print("srgan finish......")
  with torch.cuda.device(torch.cuda.current_device()): 
    torch.cuda.empty_cache()
  # re-test srgan output
  dataloader1 = DataLoader(
    ImageDataset("srgan/srgan_output", hr_shape=(1024,1024)),
    batch_size=1,
    shuffle=True,
    num_workers=0,
  )
  all_boxes_1 = [[[] for _ in range(len(dataloader1))]
               for _ in range(imdb.num_classes)]
               
  for i, gan_img in enumerate(dataloader1):
  #for i in range(len(dataloader1)):
    #gan_img = gan_output[i]
    #print(gan_img)
    arr = np.append(gan_img["origin_size"][0][0].numpy(), gan_img["origin_size"][1][0].numpy())
    gan_img_os = F.interpolate(gan_img['hr'], size=(arr[0],arr[1]), mode='bilinear')
    r = 600 / gan_img_os.shape[2]
    gan_info = np.array([[gan_img_os.shape[2], gan_img_os.shape[3], r]])
    with torch.no_grad():
      gan_img_600 = F.interpolate(gan_img_os, scale_factor=r, mode="bilinear").cuda()
      gan_info = torch.from_numpy(gan_info).cuda()
      gt_boxes
      num_boxes
    #print(gan_img.shape)
    #print(gan_info.shape)
    #print(gt_boxes)
    #print(num_boxes)
    det_tic = time.time()
    rois_1, cls_prob_1, bbox_pred_1, \
    rpn_loss_cls_1, rpn_loss_box_1, \
    RCNN_loss_cls_1, RCNN_loss_bbox_1, \
    rois_label_1 = fasterRCNN(gan_img_600, gan_info, gt_boxes, num_boxes)

    scores_1 = cls_prob_1.data
    boxes_1 = rois_1.data[:, :, 1:5]
    #print(data)
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred_1.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
          if args.class_agnostic:
              box_deltas_1 = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              box_deltas_1 = box_deltas.view(1, -1, 4)
          else:
              box_deltas_1 = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              box_deltas_1 = box_deltas.view(1, -1, 4 * len(imdb.classes))

        pred_boxes_1 = bbox_transform_inv(boxes, box_deltas_1, 1)
        pred_boxes_1 = clip_boxes(pred_boxes_1, gan_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes_1 = np.tile(boxes_1, (1, scores.shape[1]))

    pred_boxes_1 /= data[1][0][2].item()

    scores_1 = scores_1.squeeze()
    pred_boxes_1 = pred_boxes_1.squeeze()
    det_toc = time.time()
    detect_time = det_toc - det_tic
    misc_tic = time.time()
    

    ##############################   decline head-tail overlapping
    new_pred_boxes = torch.cuda.FloatTensor(300, 160).zero_()
    new_scores = torch.cuda.FloatTensor(300,40).zero_()
    for k in range(13):
      b = torch.cat((pred_boxes_1[:,12*k+4:12*k+8],pred_boxes_1[:,12*k+8:12*k+12]),0)
      s = torch.cat((scores_1[:,3*k+1],scores_1[:,3*k+2]),0)
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
    if vis:
        im = cv2.imread(imdb.image_path_at(i))
        im2show = np.copy(im)
    
    for j in range(1, imdb.num_classes):
        inds = torch.nonzero(new_scores[:,j]>thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
          cls_scores_1 = new_scores[:,j][inds]
          _, order = torch.sort(cls_scores_1, 0, True)
          if args.class_agnostic:
            cls_boxes_1 = new_pred_boxes[inds, :]
          else:
            cls_boxes_1 = new_pred_boxes[inds][:, j * 4:(j + 1) * 4]
          cls_dets_1 = torch.cat((cls_boxes_1, cls_scores_1.unsqueeze(1)), 1)
          cls_dets_1 = cls_dets_1[order]
          keep = nms(cls_boxes_1[order, :], cls_scores_1[order], cfg.TEST.NMS)
          cls_dets_1 = cls_dets_1[keep.view(-1).long()]
          
          
          if vis:
            im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
          all_boxes_1[j][i] = cls_dets.cpu().numpy()
        else:
          all_boxes_1[j][i] = empty_array

    
    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        #print(all_boxes[3][i][:,-1])
        image_scores = np.hstack([all_boxes_1[j][i][:,-1]
                                  for j in range(1, imdb.num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, imdb.num_classes):
                keep = np.where(all_boxes_1[j][i][:, -1] >= image_thresh)[0]
                all_boxes_1[j][i] = all_boxes_1[j][i][keep, :]
    
    misc_toc = time.time()
    nms_time = misc_toc - misc_tic

    sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
        .format(i + 1, len(dataloader1), detect_time, nms_time))
    sys.stdout.flush()
  
  
  
  torch.cuda.empty_cache()
  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  end = time.time()
  #print(len(all_boxes))
  #print(len(all_boxes_1[0]))
  for a in range(len(all_boxes)):
    all_boxes[a].extend(all_boxes_1[a])
    print(len(all_boxes[a]))
  print(new_indexes)
  #print(new_gt_boxes)
  imdb.evaluate_detections(all_boxes, output_dir, new_indexes, new_gt_boxes)
  
  print("test time: %0.4fs" % (end - start))
