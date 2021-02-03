# -*- coding: utf-8 -*-
"""
 @File    : grad_cam.py
 @Time    : 2020/3/14 ä¸‹å?4:06
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np


class GradCAM(object):
    """
    1: ç½‘ç?ä¸æ›´?°æ¢¯åº?è¾“å…¥?€è¦æ¢¯åº¦æ›´??    2: ä½¿ç”¨?®æ?ç±»åˆ«?„å??†å??å?ä¼ æ’­
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,?¿åº¦ä¸?
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: ç¬¬å?ä¸ªè¾¹æ¡?        :return:
        """
        self.net.zero_grad()
        output = self.net.inference([inputs])
        print(output)
        score = output[0]['instances'].scores[index]
        proposal_idx = output[0]['instances'].indices[index]  # box?¥è‡ªç¬¬å?ä¸ªproposal
        score.backward()

        gradient = self.gradient[proposal_idx].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[proposal_idx].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # ?°å€¼å?ä¸€??        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        box = output[0]['instances'].pred_boxes.tensor[index].detach().numpy().astype(np.int32)
        x1, y1, x2, y2 = box
        cam = cv2.resize(cam, (x2 - x1, y2 - y1))

        class_id = output[0]['instances'].pred_classes[index].detach().numpy()
        return cam, box, class_id


class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: ç¬¬å?ä¸ªè¾¹æ¡?        :return:
        """
        self.net.zero_grad()
        output = self.net.inference([inputs])
        print(output)
        score = output[0]['instances'].scores[index]
        proposal_idx = output[0]['instances'].indices[index]  # box?¥è‡ªç¬¬å?ä¸ªproposal
        score.backward()

        gradient = self.gradient[proposal_idx].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # ç¤ºæ€§å‡½??        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]å½’ä???        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # ?¿å??¤é›¶
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = self.feature[proposal_idx].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        # ?°å€¼å?ä¸€??        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to box scale
        box = output[0]['instances'].pred_boxes.tensor[index].detach().numpy().astype(np.int32)
        x1, y1, x2, y2 = box
        cam = cv2.resize(cam, (x2 - x1, y2 - y1))

        return cam
