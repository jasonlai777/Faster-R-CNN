# -*- coding: utf-8 -*-

import pickle
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

  
  count =0 
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
    #print(image_ids)
    nd = len(image_ids)
    #print(nd)
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
            count +=1
            R['det'][jmax] = 1
            #print(image_ids[d])
    


  return count



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.set_xlabel("Predicted Classes", fontsize=16, weight='bold')
    ax.set_ylabel("Actual Classes", fontsize=16, weight='bold')
    ax.xaxis.set_label_position('top') 
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    return im,  cax


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



          
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
                      default=0.8, type=float) 
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
  
  
def valfmt(x, pos):
    if x < 0.01:
        return "0"
    return '{:.2f}'.format(x)
## -------------------------------------------------------------
def main(classes, ignore_cls, args):
    num =0
    result_path = args.result_path
    test_file = args.test_file
    gt_file = args.gt_file
    
    cfmap = np.zeros((len(classes)+1,len(classes)))

    for i, detcls in enumerate(classes):
        if detcls == '__background__' or detcls in ignore_cls:
            continue
        det_file = result_path  +"comp4_det_"+ 'test_{}.txt'.format(detcls)
        num_sum = 0
        for j, cls in enumerate(classes):
            if cls == '__background__' :
                continue     
            num = voc_eval(
              det_file, test_file, cls, gt_file,
              ovthresh=args.ovthr, csthresh=args.csthr)
            #print(num)
            cfmap[j][i] = num
            num_sum += num
        # count background number
        with open(det_file, 'r') as f:
            lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        confidence = np.array([float(x[1]) for x in splitlines])
        keep = [c for c in confidence if c > 0.8]
        cfmap[j+1][i] = len(keep)- num_sum
        #print(cfmap[j+1][i])
        
        
    cfmap = np.delete(cfmap, 0, axis=0)
    cfmap = np.delete(cfmap, 0, axis=1)
    
    sum_of_col = np.sum(cfmap,axis = 0)
    for i in range(len(classes)-1):
      for j in range(len(classes)):
        cfmap[j][i] = round(cfmap[j][i] / sum_of_col[i], 2)
    
    fig, ax = plt.subplots(figsize=(10,10))
    #print(cfmap.shape, cfmap[1:][1:].shape)
    im, cax = heatmap(cfmap, classes[1:]+("Background",), classes[1:], ax=ax,
                       cmap='GnBu', cbarlabel="Probalility")
    texts = annotate_heatmap(im, valfmt=valfmt)
    
    fig.tight_layout()
    plt.colorbar(im, cax=cax)
    #plt.show() 
    plt.savefig('CF_matrix_T.png')      
    
    
if __name__ == '__main__':
    
    args = parse_args()
    
    args.gt_file = './data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt_annots.pkl'
    args.test_file = './data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
    args.result_path = './data/VOCdevkit2007/results/VOC2007/Main/'
    
    # classes = ('__background__',  # always index 0
    #             'A.bes(H)','A.bes(T)','A.bes','A.bic(H)','A.bic(T)','A.bic',
    #              'A.fuj(H)','A.fuj(T)','A.fuj','B.xyl(H)','B.xyl(T)','B.xyl',
    #              'C.ele(H)','C.ele(T)','C.ele','M.ent(H)','M.ent(T)','M.ent',
    #              'M.gra(H)','M.gra(T)','M.gra','M.inc(H)','M.inc(T)','M.inc',
    #              'P.cof(H)','P.cof(T)','P.cof','P.vul(H)','P.vul(T)','P.vul',
    #              'P.spe(H)','P.spe(T)','P.spe','H.sp(H)','H.sp(T)','H.sp',
    #              'M.ams(H)' ,'M.ams(T)','M.ams')###################
    
    classes = ('__background__',  # always index 0
                'A.bes(T)','A.bic(T)','A.fuj(T)','B.xyl(T)',
                'C.ele(T)','M.ent(T)','M.gra(T)','M.inc(T)',
                'P.cof(T)','P.vul(T)','P.spe(T)','H.sp(T)',
                'M.ams(T)')###################
                        
    ignore_cls = []
    
    main(classes, ignore_cls, args)
    