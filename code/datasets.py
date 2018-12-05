from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
import torch
from copy import deepcopy
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
	fimg = deepcopy(img)
	fimg_arr = np.array(fimg)
	fimg = Image.fromarray(fimg_arr)
        cimg = img.crop([x1, y1, x2, y2])

    if transform is not None:
	cimg = transform(cimg)

    
    retf = []
    retc = []
    re_cimg = transforms.Scale(imsize[1])(cimg)
    retc.append(normalize(re_cimg))

    # We use full image to get background patches

    # We resize the full image to be 126 X 126 (instead of 128 X 128)  for the full coverage of the input (full) image by 
    # the receptive fields of the final convolution layer of background discriminator
    
    my_crop_width = 126	
    re_fimg = transforms.Scale(int(my_crop_width * 76 / 64))(fimg)
    re_width, re_height = re_fimg.size

    # random cropping
    x_crop_range = re_width-my_crop_width
    y_crop_range = re_height-my_crop_width

    crop_start_x = np.random.randint(x_crop_range)	
    crop_start_y = np.random.randint(y_crop_range)

    crop_re_fimg = re_fimg.crop([crop_start_x, crop_start_y, crop_start_x + my_crop_width, crop_start_y + my_crop_width])
    warped_x1 = bbox[0] * re_width / width
    warped_y1 = bbox[1] * re_height / height
    warped_x2 = warped_x1 + (bbox[2] * re_width / width)
    warped_y2 = warped_y1 + (bbox[3] * re_height / height)

    warped_x1 =min(max(0, warped_x1 - crop_start_x), my_crop_width)
    warped_y1 =min(max(0, warped_y1 - crop_start_y), my_crop_width)
    warped_x2 =max(min(my_crop_width, warped_x2 - crop_start_x),0)
    warped_y2 =max(min(my_crop_width, warped_y2 - crop_start_y),0)

    # random flipping
    random_flag=np.random.randint(2)
    if(random_flag == 0):
	crop_re_fimg = crop_re_fimg.transpose(Image.FLIP_LEFT_RIGHT)
	flipped_x1 = my_crop_width - warped_x2
	flipped_x2 = my_crop_width - warped_x1
	warped_x1 = flipped_x1
	warped_x2 = flipped_x2

    retf.append(normalize(crop_re_fimg))
		
    warped_bbox = []
    warped_bbox.append(warped_y1)
    warped_bbox.append(warped_x1)
    warped_bbox.append(warped_y2)
    warped_bbox.append(warped_x2)

    return retf, retc, warped_bbox



class Dataset(data.Dataset):
    def __init__(self, data_dir, base_size=64, transform = None):

        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.bbox = self.load_bbox()
        self.filenames = self.load_filenames(data_dir)
        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

        
    # only used in background stage
    def load_bbox(self):
        # Returns a dictionary with image filename as 'key' and its bounding box coordinates as 'value'

        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        return filename_bbox


    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        filenames =  [fname[:-4] for fname in filenames];
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames


    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        data_dir = self.data_dir
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        fimgs, cimgs, warped_bbox = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        rand_class= random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1); # Randomly generating child code during training 
	c_code = torch.zeros([cfg.FINE_GRAINED_CATEGORIES,])
	c_code[rand_class] = 1

        return fimgs, cimgs, c_code, key, warped_bbox 

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        data_dir = self.data_dir
        c_code = self.c_code[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        _, imgs, _ = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        return imgs, c_code, key 

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)
