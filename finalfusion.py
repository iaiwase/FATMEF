import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from FATMEF import CLMEFNet


import os
from PIL import Image, ImageFile, ImageFilter
from tqdm import tqdm
import cv2
from torchvision.models import vgg16
from torchvision.models import vgg19
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def load_img(img_path):
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.open(img_path, mode='L')
    return _tensor(img).unsqueeze(0)


_tensor = transforms.ToTensor()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='model save and load')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--ue_path', type=str, default='./MEFB_dataset_gray/under-exposed/')
parser.add_argument('--oe_path', type=str, default='./MEFB_dataset_gray/over-exposed/')
parser.add_argument('--model_path', type=str,
                    default='train_result/5.11-1.pth')
parser.add_argument('--model_path2', type=str,
                    default="train_result/5.11-3.pth")
parser.add_argument('--save_path', type=str, default='./clif_/')
parser.add_argument('--resize', type=bool, default=False)
parser.add_argument('--method', type=bool, default=False)

args = parser.parse_args()

args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")

model = CLMEFNet().to(device)

state_dict = torch.load(args.model_path, map_location="cuda:0")['model']

# state_dict2 = torch.load(args.model_path2, map_location="cuda:0")['model']
# state_dict2 = {k: v for k, v in state_dict2.items() if "transformer" in k}
# print(state_dict2.keys())

if len(args.gpus) > 1:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(state_dict)
    # model.load_state_dict(state_dict2,strict=False)


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


mkdir(args.save_path)

for name in tqdm(os.listdir(args.ue_path)):
    img_path_ue = args.ue_path + name
    img_path_oe = args.oe_path + name

    model.eval()
    with torch.no_grad():
        img_ue = Image.open(img_path_ue)
        img_oe = Image.open(img_path_oe)

        img_ue = img_ue.convert('L')
        img_oe = img_oe.convert('L')

        if args.resize:
            size = 256
            img_ue = img_ue.resize((size, size))
            img_oe = img_oe.resize((size, size))

        img_ue = np.array(img_ue)
        img_oe = np.array(img_oe)

        img_ue = _tensor(img_ue).unsqueeze(0).to(device)
        img_oe = _tensor(img_oe).unsqueeze(0).to(device)

        img_ue_feats = model.encoder(img_ue.float())
        img_oe_feats = model.encoder(img_oe.float())

        if args.method==True:
            from culw import CustomModel
            vgg = vgg19(pretrained=True).features
            vgg = vgg.to(device)
            for param in vgg.parameters():
                param.requires_grad = False
            culs = CustomModel(vgg,device)
            img_ue = torch.cat((img_ue, img_ue, img_ue), 1)
            img_oe = torch.cat((img_oe, img_oe, img_oe), 1)
            s = culs(img_ue,img_oe)

            fusion_feats = (s[0]*img_ue_feats + s[1]*img_oe_feats) / 2
        else:
            fusion_feats = (img_ue_feats + img_oe_feats) / 2

        fusion_img = model.decoder(fusion_feats).squeeze(0).detach().cpu().numpy()

        fusion_img = normalization(fusion_img)

        fusion_img = fusion_img.squeeze()

        # cv2.imwrite(args.save_path+name,fusion_img*255)

        fusion_img_array = Image.fromarray((fusion_img * 255).astype(np.uint8))
        fusion_img_array.save(args.save_path + name)