
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from lib.model.utils.config import cfg
from lib.roi_data_layer.minibatch import get_minibatch, get_minibatch
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import os
from scipy.misc import imread
from lib.model.utils.config import cfg
from lib.model.utils.blob import prep_im_for_blob, im_list_to_blob

import IPython

class evaluateDataLoader(data.Dataset):
  def __init__(self, roidb, root_path, batch_size, num_classes, normalize=None):
    # use the VOCdevkit2007 roidb.......
    self._roidb = roidb
    self._num_classes = num_classes
    self.normalize = normalize
    self.batch_size = batch_size

    files = os.listdir(root_path)
    self.data_list = []
    for dir in files:
        dir_path = os.path.join(root_path, dir)
        image_path = os.listdir(dir_path)
        for im in image_path:
            self.data_list.append(os.path.join(dir_path, im))

    self.data_size = len(self.data_list)

  def __getitem__(self, index):
    im_path = self.data_list[index]
    blobs = self.get_evaluate_batch(im_path, index)
    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    # we need to random shuffle the bounding box.
    data_height, data_width = data.size(1), data.size(2)
    data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
    im_info = im_info.view(3)

    gt_boxes = torch.FloatTensor([1,1,1,1,1])
    num_boxes = 0

    # IPython.embed()
    return data, im_info, gt_boxes, num_boxes, im_path

  def get_evaluate_batch(self, im_path, index):
      # Sample random scales to use for each image in this batch

      # Get the input image blob, formatted for caffe
      # im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
      im = imread(im_path)

      if len(im.shape) == 2:
          im = im[:, :, np.newaxis]
          im = np.concatenate((im, im, im), axis=2)
      # flip the channel, since the original one using cv2
      # rgb -> bgr
      im = im[:, :, ::-1]

      target_size = cfg.TRAIN.SCALES[0]
      im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                      cfg.TRAIN.MAX_SIZE)
      im_blob = im_list_to_blob([im])
      blobs = {'data': im_blob}

      # gt boxes: (x1, y1, x2, y2, cls)
      gt_boxes = np.empty((0, 5), dtype=np.float32)
      blobs['gt_boxes'] = gt_boxes
      blobs['im_info'] = np.array([[im.shape[0], im.shape[1], im_scale]], dtype=np.float32)
      blobs['img_id'] = index
      return blobs

  def __len__(self):
    return self.data_size
