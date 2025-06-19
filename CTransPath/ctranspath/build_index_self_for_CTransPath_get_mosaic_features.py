from CTransPath_ctran import ctranspath
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms


from collections import OrderedDict
import pickle
import copy
import openslide
import multiprocessing as mp
import torch
import glob
import h5py
import numpy as np
import time
import argparse
import os
import sys

os.chdir(sys.path[0])
sys.path.append(os.getcwd())

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

script_dir = os.path.dirname(os.path.abspath(__file__))

class Mosaic_Bag_FP(torch.utils.data.Dataset):
    def __init__(self,
                 file_path,
                 wsi,
                 resolution,
                 custom_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            wsi (openslide object): Whole slide image loaded by openslide
            resolution (int): The resolution of the wsi
            custom_transforms (callable, optional): The transform to be applied on a sample
        """
        self.wsi = wsi
        self.resolution = resolution
        self.roi_transforms = custom_transforms
        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            f = h5py.File(self.file_path, "r")
            self.dset = f['coords'][:]

        self.patch_level = 0
        if self.resolution == 40:
            self.patch_size = 2048  # 512
            self.target_patch_size = 224  # 256
        elif self.resolution == 20:
            self.patch_size = 1024  # 256
            self.target_patch_size = 1024  # 256
        self.length = len(self.dset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.wsi.read_region((self.dset[idx][0], self.dset[idx][1]),
                                   self.patch_level,
                                   (self.patch_size, self.patch_size)).convert('RGB')
        img = img.resize((self.target_patch_size, self.target_patch_size))
        img = self.roi_transforms(img)
        return img, self.dset[idx]


def set_everything(seed):
    """
    Function used to set all random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_hdf5(output_path, asset_dict, mode='a'):
    """
    Function that used to store hdf5 file chunk by chunk
    """
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape,
                                       maxshape=maxshape, chunks=chunk_shape,
                                       dtype=data_type)
            dset[:] = val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val

    file.close()
    return output_path


def min_max_binarized(feat):
    """
    Min-max algorithm proposed in paper: Yottixel-An Image Search Engine for Large Archives of
    Histopathology Whole Slide Images.
    Input:
        feat (1 x 1024 np.arrya): Features from the last layer of DenseNet121.
    Output:
        output_binarized (str): A binary code of length  1024
    """
    prev = float('inf')
    output_binarized = []
    for ele in feat:
        if ele < prev:
            code = 0
            output_binarized.append(code)
        elif ele >= prev:
            code = 1
            output_binarized.append(code)
        prev = ele
    output_binarized = "".join([str(e) for e in output_binarized])
    return output_binarized


def compute_latent_features(wsi, mosaic_path, save_path, resolution, transform,
                            model, batch_size=8, num_workers=12):
    """
    Copmute the latent code of input by VQ-VAE encoder
    Input:
        wsi (openslide objet): The slide to compute
        mosaic_path (str): The path that store wsi mosaic
        save_path (str): Path to store latent code
        resolution (str): The resolution of wsi (e.g., 20x or 40x)
        transoform (torch.transforms): The transform applied to image before
        feeding into VQ-VAE
        vqvae (torch.models): VQ-VAE encoder along with codebook with weight
        from the checkpoints
        batch_size (int): The number of data processed by VQ-VAE per loop
        num_workers (int): Number of cpu used by Dataloader to load the data
    Output:
        feature list (list): list of vq-vqae latent code of length = #mosaics
        in the wsi
    """
    dataset = Mosaic_Bag_FP(mosaic_path, wsi, int(resolution[:-1]),
                            custom_transforms=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False,
                            pin_memory=True)
    count = 0
    total = len(dataset)
    if total == 0:
        return None

    mode = 'w'
    save_ctranspath_path = os.path.join(
        save_path, os.path.basename(mosaic_path))
    with torch.no_grad():
        for mosaic, coord in dataloader:
            mosaic = torch.squeeze(mosaic, 1)
            mosaic = mosaic.to(device, non_blocking=True)
            features = model(mosaic)
            features = features.cpu().numpy()
            count += features.shape[0]
            # print("Numebr of latent code processed {}/{}".format(count, total))
            asset_dict = {'features': features,
                          'coords': coord.numpy()}
            save_hdf5(save_ctranspath_path, asset_dict, mode=mode)
            mode = 'a'
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build database of FISH')
    parser.add_argument("--mosaic_path", type=str, default="./DATA/MOSAICS",
                        help="Path to mosaics")
    parser.add_argument("--slide", type=str, default="./DATA/WSI",
                        help="Path to WSIs")
    parser.add_argument('--resolution', type=str, default='40x')
    parser.add_argument("--save_path", required=True, default="./DATA/LATENT")

    args = parser.parse_args()

    # Set up cpu and device
    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 20
    pool = mp.Pool(num_workers)

    t_total_start = time.time()

    # initialize the database related object
    database = {}
    key_list = []

    print(".................... get features .................... ")

    # initialize transforms for densenet and vq-vae
    transform_ctranspath = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    # Load the Densenet
    model = ctranspath()
    model.head = torch.nn.Identity()
    td = torch.load(os.path.join(script_dir,'ctranspath.pth'))
    model.load_state_dict(td['model'], strict=True)

    model.to(device)
    model.eval()

    t_enc_start = time.time()
    mosaic_all = os.path.join(args.mosaic_path, "coord_clean", "*")
    total = len(glob.glob(mosaic_all))
    count = 0
    number_of_mosaic = 0

    for slide in [args.slide]:
        slide_id = os.path.basename(slide).replace(".ndpi", "").replace("svs","")
        mosaic_path = os.path.join(args.mosaic_path, "coord_clean", slide_id+".h5")
        # print(mosaic_path)
        try:
            t_start = time.time()
            
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

            with h5py.File(mosaic_path, 'r') as hf:
                mosaic_coord = hf['coords'][:]
            wsi = openslide.open_slide(args.slide)
            latent = compute_latent_features(wsi, mosaic_path, args.save_path,
                                             args.resolution, transform_ctranspath, model)
        except:
            print("error", slide_id)
