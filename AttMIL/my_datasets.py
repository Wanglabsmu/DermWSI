from PIL import Image
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from my_models import to_fixed_size_bag


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, svs_paths: list): #, svs_classes: list):
        self.svs_paths = svs_paths
        # self.svs_classes = svs_classes

    def __len__(self):
        return len(self.svs_paths)

    def __getitem__(self, item):
        svs_path = self.svs_paths[item]
        with h5py.File(svs_path, 'r') as hf:
            feature_map = hf['features'][:]
            coords_map = hf['coords'][:]     ##特征patches，位置信息
            feature_map_np = np.array(feature_map, dtype=np.float32)

        feature_map_tr = torch.from_numpy(feature_map_np).float()
        # [num_patch, 768]

        fixed_len_bag, orignal_bag_len, indx = to_fixed_size_bag(
            bag=feature_map_tr, bag_size=100)

        # label = self.svs_classes[item]
        coords = coords_map[indx]

        return fixed_len_bag, orignal_bag_len, coords #, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        # images, lens, labels = tuple(zip(*batch))
        images, lens,  coords= tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        lens = torch.as_tensor(lens)
        # labels = torch.as_tensor(labels)
        return images, lens, coords #, labels

