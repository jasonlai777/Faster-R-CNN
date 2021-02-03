# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os
from ... import _init_paths
import cv2
#import detectron2.data.transforms as T
from PIL import Image
from numpy import asarray
import numpy as np
import torch
#from detectron2.checkpoint import DetectionCheckpointer
#from detectron2.config import get_cfg
#from detectron2.data import MetadataCatalog
#from detectron2.data.detection_utils import read_image
#from detectron2.modeling import build_model
#from detectron2.utils.logger import setup_logger
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader



from grad_cam import GradCAM, GradCamPlusPlus
from skimage import io
from torch import nn

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_last_conv_name(net):
    """
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


class GuidedBackPropagation(object):

    def __init__(self, net):
        self.net = net
        for (name, module) in self.net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.backward_hook)
        self.net.eval()

    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """

        :param module:
        :param grad_in: tuple,
        :param grad_out: tuple,
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index:        :return:
        """
        self.net.zero_grad()
        output = self.net.inference([inputs])
        score = output[0]['instances'].scores[index]
        score.backward()

        return inputs['image'].grad  # [3,H,W]


def norm_image(image):
    """
       :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    
    :param mask: [H,W],
    :return: tuple(cam,heatmap)
    """
    # mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    #  cam = heatmap + np.float32(image)
    return norm_image(cam), heatmap


def gen_gb(grad):
    """
   uided back propagation    :param grad: tensor,[3,H,W]
    :return:
    """
    #   grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network='frcnn', output_dir='./results'):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)


def get_parser():
    parser = argparse.ArgumentParser(description="jwyang demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def main(args):
    #setup_logger(name="fvcore")
    #logger = setup_logger()
    #logger.info("Arguments: " + str(args))

    #cfg = setup_cfg(args)
    
    print(cfg)
    #build_model
    #model = build_model(cfg)
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
  
    args.cfg_file = "./../../cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
  
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
    #load weight
    #checkpointer = DetectionCheckpointer(model)
    #checkpointer.load(cfg.MODEL.WEIGHTS)
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    
    print('load model successfully!')
    
        
    #load image
    path = os.path.expanduser(args.input)
    #original_image = read_image(path, format="BGR")
    #height, width = original_image.shape[:2]
    #transform_gen = T.ResizeShortestEdge(
    #    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    #)
    #image = transform_gen.get_transform(original_image).apply_image(original_image)
    #image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(True)
    
    original_image = asarray(Image.open(path))
    height, width = original_image.shape[:2]
    image = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
    images = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

    images_iter = iter(dataloader)
    inputs = {"image": image, "height": height, "width": width}

    # Grad-CAM
    layer_name = get_last_conv_name(fasterRCNN)
    grad_cam = GradCAM(fasterRCNN, layer_name)
    mask, box, class_id = grad_cam(inputs)  # cam mask
    grad_cam.remove_handlers()

    #
    image_dict = {}
    img = original_image[..., ::-1]
    x1, y1, x2, y2 = box
    image_dict['predict_box'] = img[y1:y2, x1:x2]
    image_cam, image_dict['heatmap'] = gen_cam(img[y1:y2, x1:x2], mask)

    # Grad-CAM++
    grad_cam_plus_plus = GradCamPlusPlus(model, layer_name)
    mask_plus_plus = grad_cam_plus_plus(inputs)  # cam mask
    _, image_dict['heatmap++'] = gen_cam(img[y1:y2, x1:x2], mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # get name of classes
    meta = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    label = meta.thing_classes[class_id]

    print("label:{}".format(label))
    # # GuidedBackPropagation
    # gbp = GuidedBackPropagation(model)
    # inputs['image'].grad.zero_()  # make gradient zero
    # grad = gbp(inputs)
    # print("grad.shape:{}".format(grad.shape))
    # gb = gen_gb(grad)
    # gb = gb[y1:y2, x1:x2]
    # image_dict['gb'] = gb
    # Guided Grad-CAM
    # cam_gb = gb * mask[..., np.newaxis]
    # image_dict['cam_gb'] = norm_image(cam_gb)

    save_image(image_dict, os.path.basename(path))


if __name__ == "__main__":
    """
    Usage:export KMP_DUPLICATE_LIB_OK=TRUE
    python detection/demo.py --config-file detection/faster_rcnn_R_50_C4.yaml \
      --input ./examples/pic1.jpg \
      --opts MODEL.WEIGHTS /Users/yizuotian/pretrained_model/model_final_b1acc2.pkl MODEL.DEVICE cpu
    """
    mp.set_start_method("spawn", force=True)
    arguments = get_parser().parse_args()
    main(arguments)
