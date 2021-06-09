# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os
import _init_paths
import cv2
#import detectron2.data.transforms as T
from PIL import Image
from numpy import asarray
import numpy as np
import torch
from torch.autograd import Variable
#from detectron2.checkpoint import DetectionCheckpointer
#from detectron2.config import get_cfg
#from detectron2.data import MetadataCatalog
#from detectron2.data.detection_utils import read_image
#from detectron2.modeling import build_model
#from detectron2.utils.logger import setup_logger
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
import pprint
import time

from grad_cam import GradCAM, GradCamPlusPlus
from skimage import io
from torch import nn
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

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
            #print(layer_name)
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
    #print(heatmap.shape, image.shape)
    cam = heatmap + np.float32(image)
    return norm_image(cam), heatmap


def gen_gb(grad):
    """
   uided back propagation    :param grad: tensor,[3,H,W]
    :return:
    """
    #   grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, layer_name, j,network='frcnn', output_dir='./Grad-CAM.pytorch/results'):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}-{}-{}.jpg'.format(prefix, network, key, layer_name, j)), image)



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
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res152', type=str)
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
    return parser


def main(args):
    classes = ('__background__',  # always index 0
                         'A.bes(H)','A.bes(T)','A.bes','A.bic(H)','A.bic(T)','A.bic',
                         'A.fuj(H)','A.fuj(T)','A.fuj','B.xyl(H)','B.xyl(T)','B.xyl',
                         'C.ele(H)','C.ele(T)','C.ele','M.ent(H)','M.ent(T)','M.ent',
                         'M.gra(H)','M.gra(T)','M.gra','M.inc(H)','M.inc(T)','M.inc',
                         'P.cof(H)','P.cof(T)','P.cof','P.vul(H)','P.vul(T)','P.vul',
                         'P.spe(H)','P.spe(T)','P.spe','H.sp(H)','H.sp(T)','H.sp',
                         'M.ams(H)' ,'M.ams(T)','M.ams'                        
                         )###################
    #setup_logger(name="fvcore")
    #logger = setup_logger()
    #logger.info("Arguments: " + str(args))

    #cfg = setup_cfg(args)
    
    #print(cfg)
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
    cfg.TRAIN.USE_VERTICAL_FLIPPED = False
    cfg.TRAIN.BRIGHTNESS_CHANGE = False
    cfg.TRAIN.ROTATE_90 = False
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
      fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
      fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
      fasterRCNN = resnet(classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
      fasterRCNN = resnet(classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
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
    #print(checkpoint.keys())
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    #print(cfg.POOLING_MODE)
    print('load model successfully!')
    
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
    #load image
    #path = os.path.expanduser(args.input)
    #original_image = read_image(path, format="BGR")
    #height, width = original_image.shape[:2]
    #transform_gen = T.ResizeShortestEdge(
    #    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    #)
    #image = transform_gen.get_transform(original_image).apply_image(original_image)
    #image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(True)
    
    #original_image = asarray(Image.open(path))
    #height, width = original_image.shape[:2]
    image_set_file = os.path.join("./Grad-CAM.pytorch/test_gradcam.txt")
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    
    #print(origin_shape)

    num_images = len(imdb.image_index)    
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        len(classes), training=False, normalize = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)
                            
    fasterRCNN.eval()
    a = 0
    data_iter = iter(dataloader)
    for i in range(num_images):
      
      img_path = os.path.join("./Grad-CAM.pytorch/examples", image_index[i]+".JPG")
      img = Image.open(img_path)
      img = asarray(img)
      origin_shape = (img.shape[1],img.shape[0])
      
      
      
      data = next(data_iter)
      im_data.resize_(data[0].size()).copy_(data[0])
      im_info.resize_(data[1].size()).copy_(data[1])
      gt_boxes.resize_(data[2].size()).copy_(data[2])
      num_boxes.resize_(data[3].size()).copy_(data[3])
      im_data.requires_grad = True
         
      #info = data[1].squeeze(0).numpy()
      #inputs = {"image": data[0], "height": info[0], "width": info[1]}
      
      
      
      
      a=0
      for (layer_name, module) in fasterRCNN.named_modules():
        
        if isinstance(module, nn.Conv2d) and layer_name[-5:-1] =="conv":
          #print(layer_name)
        
        #if isinstance(module, nn.Conv2d) and layer_name =="RCNN_base.6.22.conv2":
          #Grad-CAM
          #get_last_conv_name(fasterRCNN)
          #layer_name = "RCNN_base.6.15.conv3"
          #module = []
          #print(layer_name, module)
          grad_cam = GradCAM(fasterRCNN, layer_name, module)
          mask, box, class_id, cam_full = grad_cam(data, im_data, im_info, gt_boxes, num_boxes, args, imdb, origin_shape)  # cam mask
          cam_full = cv2.normalize(cam_full, None, 0, 255, cv2.NORM_MINMAX)
          #print(cam_full)
          heatmap_full = cv2.applyColorMap(np.uint8(cam_full), cv2.COLORMAP_JET)
          #heatmap_full = np.float32(heatmap_full) / 255
          #heatmap_full = heatmap_full[..., ::-1]#rgb ->  bgr
          #print(heatmap_full.shape)
          #cv2.imwrite("%s_full_img.jpg"%(layer_name), heatmap_full)
          
          grad_cam.remove_handlers()
          #print(len(mask))
          for j in range(len(mask)):
            
            image_dict = {}
            x1, y1, x2, y2 = box[j]
      
            image_dict['predict_box'] = img[y1:y2, x1:x2]
            image_cam, image_dict['heatmap'] = gen_cam(img[y1:y2, x1:x2], mask[j])
        
            # Grad-CAM++
            #grad_cam_plus_plus = GradCamPlusPlus(model, layer_name)
            #mask_plus_plus = grad_cam_plus_plus(inputs)  # cam mask
            #_, image_dict['heatmap++'] = gen_cam(img[y1:y2, x1:x2], mask_plus_plus)
            #grad_cam_plus_plus.remove_handlers()
        
            # get name of classes
            #meta = MetadataCatalog.get(
            #    cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
            #)
            label = classes[class_id]
        
            print("label:{}".format(label))
            #print("box %d"%j)
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
            save_image(image_dict, os.path.basename(img_path),layer_name, j)
            a+=1
            print("image count: %d, layers count:%d"%(i,a))
            
      im_data.requires_grad = False

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
