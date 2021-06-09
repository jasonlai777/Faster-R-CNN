from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib as mpl

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete



class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        '''
        self._classes = ('__background__',  # always index 0
                         'A.bes','A.bic',
                         'A.fuj','B.xyl',
                         'C.ele','M.ent',
                         'M.gra','M.inc',
                         'P.cof','P.vul',
                         'P.spe','H.sp',
                         'M.ams'                    
                         )###################
        '''
        self._classes = ('__background__',  # always index 0
                         'A.bes(H)','A.bes(T)','A.bes','A.bic(H)','A.bic(T)','A.bic',
                         'A.fuj(H)','A.fuj(T)','A.fuj','B.xyl(H)','B.xyl(T)','B.xyl',
                         'C.ele(H)','C.ele(T)','C.ele','M.ent(H)','M.ent(T)','M.ent',
                         'M.gra(H)','M.gra(T)','M.gra','M.inc(H)','M.inc(T)','M.inc',
                         'P.cof(H)','P.cof(T)','P.cof','P.vul(H)','P.vul(T)','P.vul',
                         'P.spe(H)','P.spe(T)','P.spe','H.sp(H)','H.sp(T)','H.sp',
                         'M.ams(H)','M.ams(T)','M.ams')
        
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.JPG'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',index + self._image_ext)
        #image_path = os.path.join("./Grad-CAM.pytorch/examples", index + self._image_ext)#######for grad cam
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        #self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',self._image_set + '.txt')
        #image_set_file = os.path.join("./Grad-CAM.pytorch/test_gradcam.txt")#######for grad cam
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
            #print(len(image_index))
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = pickle.load(fid)
        #     print('{} gt roidb loaded from {}'.format(self.name, cache_file))
        #     return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        # with open(cache_file, 'wb') as fid:
        #     pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        # print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        '''
        temp_objs = []
        for obj in objs:
            if obj.find('name').text.strip()== 'A.bes' or\
             obj.find('name').text.strip()== 'A.bic' or\
             obj.find('name').text.strip()== 'A.fuj' or\
             obj.find('name').text.strip()== 'B.xyl' or\
             obj.find('name').text.strip()== 'C.ele' or\
             obj.find('name').text.strip()== 'M.ent' or\
             obj.find('name').text.strip()== 'M.gra' or\
             obj.find('name').text.strip()== 'M.inc' or\
             obj.find('name').text.strip()== 'P.cof' or\
             obj.find('name').text.strip()== 'P.spe' or\
             obj.find('name').text.strip()== 'P.vul' or\
             obj.find('name').text.strip()== 'H.sp' or\
             obj.find('name').text.strip()== 'M.ams':
               temp_objs.append(obj)
        objs = temp_objs
        '''
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult
            
            cls = self._class_to_ind[obj.find('name').text.strip()]#lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'vflipped':False,
                'brightness':False,
                'rotate90':False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes, new_indexes):##########
        test_with_srgan_flag = True
        if test_with_srgan_flag == True:
            self.image_index.extend(new_indexes)
            print(len(self.image_index))  
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    #print(cls_ind, im_ind)                    
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
    def _do_python_eval(self, new_indexes, new_gt_boxes, output_dir='output'):###########
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile= os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        f1s = []
        ntps = 0
        nfps = 0
        nfns = 0
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        weights = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            #if cls[-1] == ")":
            filename = self._get_voc_results_file_template().format(cls)
            detfile = filename.format(cls)
            with open(detfile, 'r') as f:
              lines = f.readlines()            
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            #print(image_ids)
            weights.append(len(image_ids))
            rec, prec, ap, f1, ntp, nfp, nfn = voc_eval(
              filename, annopath, imagesetfile, cls, cachedir, new_indexes, new_gt_boxes, ovthresh=0.5,
              use_07_metric=use_07_metric)
            f1s += [f1]
            aps += [ap]
            ntps += ntp
            nfps += nfp
            nfns += nfn
            #if cls == "H.sp":
              #print(prec, rec)
            pl.plot(rec, prec, lw=2,
                    label='{} (AP = {:.4f})'
                          ''.format(cls, ap))
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        
        accuracy = (ntps+0)/(ntps+0+nfps+nfns)        
        #mpl.rcParams['xtick.direction'] = 'in'
        #mpl.rcParams['ytick.direction'] = 'in'
        pl.tick_params(axis= 'x', direction='in', labelsize = 18)
        pl.tick_params(axis= 'y', direction='in', labelsize = 18)
        pl.xlabel('Recall', fontsize=18)
        pl.ylabel('Precision', fontsize=18)
        pl.grid(False)
        pl.ylim([0.0, 1.05])
        pl.xlim([0.0, 1.05])
        
        #print(weights)
        #plt.title('mAP = %.4f' % np.average(aps, weights = weights), fontsize=20)
        pl.title('mAP = %.4f' % np.mean(aps), fontsize=20)
        pl.legend(loc="best")     
        plt.savefig("PR_curve.png")
        #plt.show()
        
        
        
        #print('Mean AP = {:.4f}'.format(np.average(aps, weights = weights)))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('CS threshold = 0.8')
        print('Accuracy = %.4f'%(accuracy))
        print("f1-score for all classes: %f"%(np.mean(f1s)))
        #print('~~~~~~~~')
        #print('Results:')
        #for ap in aps:
            #print('{:.3f}'.format(ap))
        #print('{:.3f}'.format(np.mean(aps)))
        #print('~~~~~~~~')        
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir, new_indexes, new_gt_boxes):#####new_indexes, new_gt_boxes for srgan
        self._write_voc_results_file(all_boxes, new_indexes)
        self._do_python_eval(new_indexes, new_gt_boxes, output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed;

    embed()
