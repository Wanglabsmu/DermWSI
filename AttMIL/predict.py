from sklearn.metrics import classification_report, confusion_matrix, precision_score, roc_curve, auc
from plotUMAP import plot_umap, fit_classifier
from tqdm import tqdm
from my_models import MILModel
from my_datasets import MyDataSet
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import sys
import os
from itertools import cycle
from sklearn.preprocessing import label_binarize
from scipy import interp
from PIL import Image, ImageDraw
import openslide


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir(sys.path[0])
# from torch.utils.tensorboard import SummaryWriter
script_dir = os.path.dirname(os.path.abspath(__file__))


def draw_markers(wsi, marker_size, positions, color=(0, 0, 0)):
    # wsi = openslide.open_slide(wsi)#openslide.OpenSlide(wsi_path)
    dimensions = wsi.level_dimensions[0]
    image = wsi.read_region((0, 0), 0, dimensions).convert("RGB")

    draw = ImageDraw.Draw(image)
    for pos in positions:
        x, y = pos
        draw.rectangle([x, y, x + marker_size, y + marker_size], outline=color, width=20)

    return image

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dict_label = {0:'ED', 1: 'LP', 2: 'MF', 3: 'PR', 4:'PRP', 5:'Pso', 6: 'SLE'}


    num_classes = len(dict_label)


    # 实例化验证数据集
    test_dataset = MyDataSet(svs_paths=[args.feature_path])

    batch_size = args.batch_size
    # number of workers
    nw = 0
    # print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_dataset.collate_fn)

    model = MILModel(n_feats=768, n_out=num_classes, with_attention_scores=True) #768
    # print("模型总参数:", sum(p.numel() for p in model.parameters()))

    model.load_state_dict(torch.load(os.path.join(script_dir,args.model_weights),
                          map_location='cpu'), strict=False)
    model = model.to(device)

    model.eval()



    with torch.no_grad():
        #data_loader = tqdm(test_loader, file=sys.stdout)
        for step, data in enumerate(test_loader):
            images, lens, coords = data
            # print(images)
            # print(lens)
            images = images.to(device)
            lens = lens.to(device)
            pred, scores = model((images, lens))
            pred_classes = torch.max(pred, dim=1)[1] ##预测的结果
            pre_label = dict_label[pred_classes.cpu().detach().numpy()[0]]
            if pre_label == "SLE":
                pre_label = "LE"
            print(f"\nPredicted Label: {pre_label}")
            patch_coords = np.array(coords[0])
            scores = scores.detach().cpu().squeeze().numpy().flatten()   # 转为numpy并展平为一维数组
            scores = scores[:patch_coords.shape[0]] 
            df = pd.DataFrame({
                'score': scores,
                'coordinates': list(map(tuple, patch_coords))  # 用元组存储坐标对
                })# 处理score张量
            df.sort_values(by='score',ascending=False,inplace=True)
            sub_df = df.iloc[:int(df.shape[0]*0.1),]

            wsi = openslide.open_slide(args.wsi_path)
            sample = os.path.basename(args.wsi_path).replace(".ndpi","").replace(".svs","")

            if not os.path.exists(args.out+"/"+sample):
                os.makedirs(args.out+"/"+sample)
            
            marked_image = draw_markers(wsi, 2048, sub_df['coordinates'].tolist())
            marked_image.save(args.out+"/"+sample+'/'+sample+"_marked.jpg") 

            for item in sub_df['coordinates'].tolist():
                # print(item)
                x,y=item
                region = wsi.read_region((x, y), 0, (2048, 2048)).convert('RGB')
                region.save(args.out+"/"+sample+'/'+str(x)+"_"+str(y)+'.jpg')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights', type=str, default="model_checkpoint_best.pth")
    parser.add_argument('--wsi_path', type=str)
    parser.add_argument('--feature_path', type=str)
    parser.add_argument('--batch-size', type=int, default=1)#64)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--split_batch', type=bool, default=False)
    parser.add_argument('--out', type=str, default="OUTPUT")
    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
