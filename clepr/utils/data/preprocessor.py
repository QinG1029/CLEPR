from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import copy
from tqdm import tqdm
class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, mutual=False,cache_in_memory=True,contrastive_learning=False,random_state=42):
        super(Preprocessor, self).__init__()
        assert not (mutual and contrastive_learning), "mutual and contrastive_learning can not be True at the same time"
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mutual = mutual
        self.contrastive_learning=contrastive_learning
        self.__cache_in_memory = cache_in_memory
        self.__cache_in_memory = False #WARNING: 只推荐在数据集较小的情况下使用，或者内存足够大的情况下使用
        self.__image_cache={}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        if self.contrastive_learning:
            return self._get_contrastive_learning_item(indices)
        return self._get_single_item(indices)
    
    def __get_image(self, fpath:str)->Image.Image:
        img=None
        if fpath in self.__image_cache:
            img=copy.deepcopy(self.__image_cache[fpath])
        else:
            img = Image.open(fpath).convert('RGB')
            if self.__cache_in_memory:
                self.__image_cache[fpath]=copy.deepcopy(img)
        return img

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        
        img=self.__get_image(fpath)

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid

    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        
        img_1=self.__get_image(fpath)
        
        img_2 = img_1.copy()

        if self.transform is not None:
            img_1 = self.transform(img_1)
            #img_2 = self.transform(img_2)
            img_2 =copy.deepcopy(img_1)#减少重复的计算

        return img_1, img_2, pid
    
    def _get_contrastive_learning_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img_1=self.__get_image(fpath)
        if self.transform is not None:
            img_1 = self.transform(img_1)
        return img_1, pid
