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

def cal_confusion(data, img_name, partial_flag = False):
  # data = {"class1": [[x1,y1,x2,y2,score],[],...], "class2": [[]..]..} (for all)
  # data = {"class1": [x1,y1,x2,y2,score], "class2": [], ..} (for full)
  acc = 0
  TP_FP_FN = 0
  count_all_pdt = 0
  counter = 0
  path_of_xml = "/home/ubuntu/Faster-R-CNN/data/VOCdevkit2007/VOC2007/Annotations/"
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
    if gt_classes[ix][:n] in data.keys():
      #print(data[gt_classes[ix]], boxes[ix, :])
      if iou(data[gt_classes[ix][:n]], boxes[ix, :]) > 0.5 \
         and check_overlap(data[gt_classes[ix][:n]], boxes[ix, :])\
         and not partial_flag :## if full-length label, all labels true
        acc = 3
        counter = 3
        break
      elif iou(data[gt_classes[ix][:n]], boxes[ix, :]) > 0.5 \
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
    else:
      counter+=1
      continue
  TP_FP_FN+=counter 
  data_values = list(data.values())
  for i in range(len(data_values)):
    count_all_pdt = count_all_pdt + len(data_values[i])
  
  TP = acc
  return TP, TP_FP_FN


def cal_accuracy_with_weights(voting_results, group_count, imgname, H_w, T_w, FL_w):

    # calculate the score for each nematode  
    groups = list(voting_results.keys())
    datas = list(voting_results.values()) 
    #print("gresult: "+str(voting_results))
    results = {}
    partial_flag = False
    if group_count != 0:
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
            results[s][4] = results[s][4] + scores[m]*H_w
          elif species[m][-2] == "T":
            results[s][4] = results[s][4] + scores[m]*T_w
          else:
            #print(scores[m])
            results[s][4] = results[s][4] + scores[m]*FL_w
        
        #print("vresult: "+str(results))
                            
        list_of_species = list(results.keys())
        list_of_scores = list(results.values())
        #print(list_of_scores) 
        highest_data = {}
        highest_score = [0,0,0,0,0]
        highest_class = ""
        for m in range(len(list_of_species)):# find the highest score class 
          #print(list_of_scores[m], highest_score)           
          if float(list_of_scores[m][4])>highest_score[4]:
            
#            if im_in.shape == (4000,6000,3):
#              highest_score = list_of_scores[m]
#            else:
#              highest_score = [list_of_scores[m][1],4000-list_of_scores[m][2],list_of_scores[m][3],4000-list_of_scores[m][0], list_of_scores[m][4]]
#            
            highest_score = list_of_scores[m]
            highest_class = list_of_species[m]
            
        highest_data[highest_class] = highest_score
        TP2, TP_FP_FN2 = cal_confusion(highest_data, imgname,  partial_flag)

        return TP2, TP_FP_FN2
    return 0, 0

def voting(voting_data):
    voting_results = {}
    list_of_keys = list(voting_data.keys())
    list_of_values = list(voting_data.values())
    #print(list_of_keys,list_of_values)
    # initial flag
    if voting_data != {}:
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

      return voting_results, group_count
    else:
      return voting_results, 0

    
if __name__ == '__main__':




  imglist_dir = '/home/ubuntu/Faster-R-CNN/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
  det_dir = '/home/ubuntu/Faster-R-CNN/data/VOCdevkit2007/VOC2007/Annotations'

  pascal_classes = np.asarray(['__background__',  # always index 0
                         'A.bes(H)','A.bes(T)','A.bes','A.bic(H)','A.bic(T)','A.bic',
                         'A.fuj(H)','A.fuj(T)','A.fuj','B.xyl(H)','B.xyl(T)','B.xyl',
                         'C.ele(H)','C.ele(T)','C.ele','M.ent(H)','M.ent(T)','M.ent',
                         'M.gra(H)','M.gra(T)','M.gra','M.inc(H)','M.inc(T)','M.inc',
                         'P.cof(H)','P.cof(T)','P.cof','P.vul(H)','P.vul(T)','P.vul',
                         'P.spe(H)','P.spe(T)','P.spe','H.sp(H)','H.sp(T)','H.sp',
                         'M.ams(H)' ,'M.ams(T)','M.ams'])###################
  imglist= []
  with open(imglist_dir) as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip("\n")
      imglist.append(line)
  #print(imglist)

  result_path = "data/VOCdevkit2007/results/VOC2007/Main/"
  c = 0
  H_w = 0.05
  T_w = 0.25
  FL_w = 0.55
  acc = 0
  highest_acc = 0
  highest_wpair = [0,0,0]
  
  while H_w < 1.00:
    if round(H_w, 2) >= 0.15:
      break 
    while T_w < 1.00:
      if round(T_w, 2) >= 0.35:
        break
      while FL_w < 1.00:
        if round(FL_w, 2) >= 0.65:
          break
        if round(H_w+T_w+FL_w, 2) == 1.00:
          print("weighted pair: (%.2f,%.2f,%.2f)"%(H_w,T_w,FL_w))
          TP1s = 0
          TP2s = 0
          TP_FP_FN1s = 0
          TP_FP_FN2s = 0
          num_images = len(imglist)
          img_count = 0
          while (img_count < num_images):
            voting_data = {}
            for i, detcls in enumerate(pascal_classes):
              if detcls == '__background__' :
                  continue
              det_file = result_path  +"comp4_det_"+ 'test_{}.txt'.format(detcls)
              #print(det_file)
              with open(det_file) as f:
                lines = f.readlines()
                for line in lines:
                  items = line.strip("\n").split(" ")
                  #items = [flaot(i) for i in items]
                  if items[0] == imglist[img_count] and float(items[1]) > 0.5:
                    if detcls not in voting_data.keys():
                      voting_data[detcls] = [[int(float(items[2])),int(float(items[3])),int(float(items[4])),\
                                              int(float(items[5])),float(items[1])]]
                    else:
                      voting_data[detcls].append([int(float(items[2])),int(float(items[3])),\
                                                  int(float(items[4])),int(float(items[5])),float(items[1])])
                      #print("appppppppppp:" + str(voting_data))
              #print(voting_data)    
              #exit()

            #print(voting_data)  
            voting_results, group_count = voting(voting_data)
            #print(voting_results)
            TP2, TP_FP_FN2 = cal_accuracy_with_weights(voting_results, group_count, imglist[img_count], H_w, T_w, FL_w)
            TP2s += TP2
            TP_FP_FN2s += TP_FP_FN2
            sys.stdout.write('im_detect: {:d}/{:d}\r' \
                              .format(img_count, num_images))
            sys.stdout.flush()
            img_count+=1
          
          
          acc = TP2s/TP_FP_FN2s
          if acc > highest_acc:
              highest_acc = acc
              highest_wpair = [round(H_w,2), round(T_w,2), round(FL_w,2)]
              print(round(highest_acc,4), highest_wpair) 
        FL_w +=0.01
      
      T_w+=0.01
      FL_w = 0
    
    H_w+=0.01
    T_w = 0
    FL_w = 0
          
          
  

  print("====================================================")
  print("Highest accuracy is %.4f"%(highest_acc))
  print("With the weight pair (Head, Tail, Full-length): (%.2f, %.2f, %.2f)"\
        %(highest_wpair[0],highest_wpair[1],highest_wpair[2]))
  print("====================================================")

      

  




      
