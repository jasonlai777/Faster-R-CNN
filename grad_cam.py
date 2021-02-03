# -*- coding: utf-8 -*-
"""
 @File    : grad_cam.py
 @Time    : 
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np
import _init_paths
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import torch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.roi_layers import nms

class GradCAM(object):
    """
    """

    def __init__(self, net, layer_name, module):
        self.net = net
        self.layer_name = layer_name
        self.module = module##################
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input_feat, output_feat):
        self.feature = output_feat
        print("feature shape:{}".format(output_feat.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,
        :return:
        """
        #print("1111")
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            #print(name, module)
            if name == self.layer_name:
            #if name == "RCNN_base.6.8.conv3":
                self.handlers.append(self.module.register_forward_hook(self._get_features_hook))
                self.handlers.append(self.module.register_backward_hook(self._get_grads_hook))
    def remove_handlers(self):
        for handle in self.handlers:
            #handle.remove()
            del handle
            
    
    def bbox_reg(self, rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label,args,imdb,scores,boxes,im_info,data):
      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred
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
          pred_boxes = clip_boxes(pred_boxes, im_info, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      return scores, pred_boxes

    def __call__(self, data, im_data, im_info, gt_boxes, num_boxes, args, imdb, origin_shape, index=0):
        """
        :param inputs: im_data, im_info, gt_boxes, num_boxes (jwyang)
        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index:       :return:
        """
        self.net.zero_grad()
        #output = self.net.inference([inputs])
        outputs = self.net(im_data, im_info, gt_boxes, num_boxes)
        ##outputs: (rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label)
        #print(len(outputs))
        scores = outputs[1]
        boxes = outputs[0][:, :, 1:5]
        scores, pred_boxes = self.bbox_reg(outputs[0],outputs[1],outputs[2],outputs[3],outputs[4],outputs[5],outputs[6],outputs[7],args, imdb, scores, boxes, im_info, data)
        
        #print(scores.shape, pred_boxes.shape)
        thresh = 0.8
        store_max = 0
        #print(len(scores[0]))
        for j in range(1, imdb.num_classes):
          
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0 and imdb.classes[j][-1] !=")":
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]              
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            if cls_scores[order[0]] > store_max:
              store_max = cls_scores[order[0]]
              keep_class = j           
              cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
              # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
              cls_dets = cls_dets[order]
              keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
              cls_dets = cls_dets[keep.view(-1).long()]
              keep_bidx = inds[order[keep]].cpu().numpy()#.item()
          '''
          if imdb.classes[j] == "A.fuj":
            cls_scores = scores[:,j]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[:, :]              
            else:
              cls_boxes = pred_boxes[:][:, j * 4:(j + 1) * 4]
            if cls_scores[order[0]] > store_max:
              store_max = cls_scores[order[0]]
              keep_class = j           
              cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
              # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
              cls_dets = cls_dets[order]
              keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
              cls_dets = cls_dets[keep.view(-1).long()]
              keep_bidx = order[keep].cpu().numpy()
              #print(keep_bidx)
          '''
        #keep_bidx = [keep_bidx[0]]
        
        cam_crops = []
        boxes = []
        #print(len(keep_bidx))
        for i in range(len(keep_bidx)):
          score = scores[keep_bidx[i], keep_class]
          score.backward(retain_graph=True)
          #print(score)
          #print(self.gradient[0])
          gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]         
          weight = np.mean(gradient, axis=(1, 2))  # [C]
          feature = self.feature[0].cpu().data.numpy()  # [C,H,W]
          #print(feature.shape)
          cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
          
          box = cls_dets[i][:4].detach().cpu().numpy().astype(np.int32)
          x1, y1, x2, y2 = box
          '''
          ###collect each gradient of single cam 
          for k in range(cam.shape[0]):
            f_origin = cv2.resize(feature[k,...], origin_shape)
            f_crop = f_origin[y1:y2, x1:x2]
            heatmap = cv2.applyColorMap(np.uint8(255 * f_crop), cv2.COLORMAP_JET)
            
            #heatmap = np.float32(heatmap) / 255
            #heatmap = heatmap[..., ::-1]  # bgr to rgb
            print(k)
            cv2.imwrite("cam_img/%d_%d.jpg"%(i, k), heatmap)
          ''' 
            
          cam = np.sum(cam, axis=0)  # [H,W]
          cam = np.maximum(cam, 0)  # ReLU
  
          #     cam -= np.min(cam)
          cam /= np.max(cam)          
          #print(x2-x1,y2-y1)
          cam_full = cv2.resize(cam, origin_shape)
          cam_crop = cam_full[y1:y2, x1:x2]
          cam_crops.append(cam_crop)
          boxes.append(box)
          class_id = keep_class
          
        return cam_crops, boxes, class_id, cam_full
        




class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index:  :return:
        """
        self.net.zero_grad()
        output = self.net.inference([inputs])
        print(output)
        score = output[0]['instances'].scores[index]
        proposal_idx = output[0]['instances'].indices[index]  # 
        score.backward()

        gradient = self.gradient[proposal_idx].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  
        norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = self.feature[proposal_idx].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        #     cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to box scale
        box = output[0]['instances'].pred_boxes.tensor[index].detach().numpy().astype(np.int32)
        x1, y1, x2, y2 = box
        cam = cv2.resize(cam, (x2 - x1, y2 - y1))

        return cam
