import os
import sys
import re
import six
import math
import lmdb
import torch

import pandas as pd
from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img



class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        transform = ResizeNormalize((self.imgW, self.imgH))
        image_tensors = [transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


class custom_dataset(Dataset):
    def __init__(self,data_csv_file_path):
        df = pd.read_csv(data_csv_file_path,encoding="cp949")
        self.image_path_list = df['path_file'].to_list()
        self.text_label_list = df['label'].to_list()
        self.nSamples = len(self.image_path_list)
    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index]).convert('L')
        return (img, self.text_label_list[index])

    
class custom_Batch_Balanced_Dataset(object):
    def __init__(self, opt):
        self.dataloader_iter_list = []
        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        _dataset = custom_dataset("/data/work_dir/img/generate_text_ko/mrjaehong_text_generation/generate_img/label.csv")
        _data_loader = torch.utils.data.DataLoader(_dataset, batch_size=opt.batch_size,shuffle=True,num_workers=int(opt.workers),collate_fn=_AlignCollate, pin_memory=True)
     
        self.dataloader_iter_list.append(iter(_data_loader))
    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            image, text = data_loader_iter.next()
            balanced_batch_images.append(image)
            balanced_batch_texts += text


        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts

    
    
