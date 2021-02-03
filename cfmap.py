# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:10:42 2019

@author: Chun
"""

import pickle
import numpy as np
import csv
import argparse


## -------------------------------------------------------------
'''
def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects
'''
## -------------------------------------------------------------
def voc_ap(rec, prec):
  """
  ap = voc_ap(rec, prec)
  Compute VOC AP given precision and recall.
  """
  # correct AP calculation
  # first append sentinel values at the end
  mrec = np.concatenate(([0.], rec, [1.]))
  mpre = np.concatenate(([0.], prec, [0.]))
  # compute the precision envelope
  for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
  # to calculate area under PR curve, look for points
  # where X axis (recall) changes value
  i = np.where(mrec[1:] != mrec[:-1])[0]

  # and sum (\Delta recall) * prec
  ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  
  return ap

## -------------------------------------------------------------

def voc_eval(detpath,
             imagesetfile,
             classname,
             cachefile,
             ovthresh=0.5,
             csthresh=0.05):
  """rec, prec, ap = voc_eval(detpath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              csthresh)

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  with open(cachefile, 'rb') as f:
    recs = pickle.load(f)

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

  # read dets
  
  with open(detpath, 'r') as f:
    lines = f.readlines()

  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    keep_score = [sorted_scores < -csthresh]
    sorted_ind = sorted_ind[keep_score]
#    for sc in sorted_scores:
#        print(sorted_scores.shape)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
          else:
            fp[d] = 1.
      else:
        fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#  print(np.maximum(tp + fp, np.finfo(np.float64).eps))
  ap = voc_ap(rec, prec)

  return rec, prec, ap, tp[-1], keep_score



def dict2csv(cfmap, classes, ignore_cls, filename, use_default_cls=False):
    # ?��? confusion matrix ?��?�?, ?��?容�?誤判?��??�放一�?
    # 如�??�要�???use_default_cls = True
    if use_default_cls:
        fieldnames = list(classes)
        fieldnames[0] = 'class'
    else:
        fieldnames = ['class',
                       'A.bes(H)','A.bes(T)','A.bes','A.bic(H)','A.bic(T)','A.bic',
                       'A.fuj(H)','A.fuj(T)','A.fuj','B.xyl(H)','B.xyl(T)','B.xyl',
                       'C.ele(H)','C.ele(T)','C.ele','M.ent(H)','M.ent(T)','M.ent',
                       'M.gra(H)','M.gra(T)','M.gra','M.inc(H)','M.inc(T)','M.inc',
                       'P.cof(H)','P.cof(T)','P.cof','P.vul(H)','P.vul(T)','P.vul',
                       'P.spe(H)','P.spe(T)','P.spe','H.sp(H)','H.sp(T)','H.sp',
                       'M.ams(H)' ,'M.ams(T)','M.ams'                         
                       ]###################

    with open(filename, 'w', newline='') as csvfile:
      # 定義欄�?
      
      # �?dictionary 寫入 CSV �?
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
      # 寫入第�??��?欄�??�稱
      writer.writeheader()
      for cls in fieldnames:
          if cls in ignore_cls or cls == 'class':
              continue
          cfmap[cls]['class'] = cls
          # 寫入資�?
          writer.writerow(cfmap[cls])
          
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--use_voc_class', dest='use_default_cls',
                      help='whether sort classes like voc',
                      action='store_true') 
  parser.add_argument('--ovthr', dest='ovthr',
                      help='IoU threshold',
                      default=0.5, type=float) 
  parser.add_argument('--csthr', dest='csthr',
                      help='confidence score threshold',
                      default=0.0, type=float) 
  parser.add_argument('--out', dest='out_file',
                      help='save cfmap file',
                      default='cfmap.csv', type=str)
  parser.add_argument('--gt_file', dest='gt_file',
                      help='ground truth pickle file',
                      default='', type=str)
  parser.add_argument('--test_file', dest='test_file',
                      help='VOC test.txt path',
                      default='', type=str)
  parser.add_argument('--result_path', dest='result_path',
                      help='result detect txt file dir path',
                      default='', type=str)
  args = parser.parse_args()
    
  return args
## -------------------------------------------------------------
def main(classes, ignore_cls, args):
    
    result_path = args.result_path
    test_file = args.test_file
    gt_file = args.gt_file
    
    cfmap = {}
    aps = []
    for j, detcls in enumerate(classes):
        if detcls == '__background__' or detcls in ignore_cls:
                continue
        cfmap[detcls] = {}
        det_file = result_path  +"comp4_det_"+ 'test_{}.txt'.format(detcls)
        for i, cls in enumerate(classes):
            if cls == '__background__' :
                continue
        #    filename = get_voc_results_file_template().format(cls)
            
            rec, prec, ap, tp, sc = voc_eval(
                det_file, test_file, cls, gt_file,
                ovthresh=args.ovthr, csthresh=args.csthr)
            aps += [ap]
            cfmap[detcls][cls] = ap
            
    dict2csv(cfmap, classes, ignore_cls, args.out_file, args.use_default_cls)
    #        print('detect {} but {} = {:d}/{:d}'.format(detcls, cls, int(tp), ndets))
    #        with open(os.path.join('./', cls + '_pr.pkl'), 'wb') as f:
    #            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    
if __name__ == '__main__':
    
    args = parse_args()
    
    args.gt_file = './data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt_annots.pkl'
    args.test_file = './data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
    args.result_path = './data/VOCdevkit2007/results/VOC2007/Main/'
    
    classes = ('__background__',  # always index 0
                'A.bes(H)','A.bes(T)','A.bes','A.bic(H)','A.bic(T)','A.bic',
                 'A.fuj(H)','A.fuj(T)','A.fuj','B.xyl(H)','B.xyl(T)','B.xyl',
                 'C.ele(H)','C.ele(T)','C.ele','M.ent(H)','M.ent(T)','M.ent',
                 'M.gra(H)','M.gra(T)','M.gra','M.inc(H)','M.inc(T)','M.inc',
                 'P.cof(H)','P.cof(T)','P.cof','P.vul(H)','P.vul(T)','P.vul',
                 'P.spe(H)','P.spe(T)','P.spe','H.sp(H)','H.sp(T)','H.sp',
                 'M.ams(H)' ,'M.ams(T)','M.ams')###################
                         
    ignore_cls = []
    
    main(classes, ignore_cls, args)
    