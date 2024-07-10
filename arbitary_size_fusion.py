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

import cv2
import argparse
from collections import OrderedDict
import string
from glob import glob
import os
from PIL import Image, ImageFile
import torch
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True
_tensor = transforms.ToTensor()
_pil_rgb = transforms.ToPILImage('RGB')
_pil_gray = transforms.ToPILImage()
device = 'cuda'


def get_block(img, block_size=256):
    '''
    The original image is cut into blocks according to block_size
    output: blocks [blocks_num, block_size, block_size]
    '''
    blocks = []
    m, n = img.shape
    # print(type(img))
    # print(img.shape)
    img_pad = np.pad(img, ((0, 256 - m % block_size), (0, 256 - n % block_size)), mode='reflect')  # mirror padding
    m_block = int(np.ceil(m / block_size))  # Calculate the total number of blocks
    n_block = int(np.ceil(n / block_size))  # Calculate the total number of blocks

    # cutting
    for i in range(0, m_block):
        for j in range(0, n_block):
            block = img_pad[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size]
            blocks.append(block)
    blocks = np.array(blocks)
    return blocks


def fuse(img1, img2):
    '''
    block fusion
    '''
    block_num = img1.shape[0]

    final_fusion = np.zeros_like(img1,dtype = float)

    for i in range(block_num):
        img1_inblock = _tensor(img1[i, :, :]).unsqueeze(0).to(device)
        img2_inblock = _tensor(img2[i, :, :]).unsqueeze(0).to(device)
        # plt.figure()
        # plt.title('ori_img1')
        # plt.imshow(img1[i, :, :], cmap='gray')
        # plt.show()
        img_fusion = fusion(x1=img1_inblock, x2=img2_inblock, model=model)
        # plt.figure()
        # plt.title('img_fusion[0]')
        # plt.imshow(img_fusion[0], cmap='gray')
        # plt.show()
        # note that no normalization should be used in different block fusion
        # img_fusion = MaxMinNormalization(img_fusion[0], torch.max(img_fusion[0]), torch.min(img_fusion[0]))

        # img_fusion = _pil_gray(img_fusion)
        # img_fusion = np.asarray(img_fusion)
        # print(img_fusion)

        img_fusion = img_fusion.numpy()
        # img_fusion = normalization(img_fusion)
        img_fusion = img_fusion.squeeze()
        # print(img_fusion)
        # img_fusion = Image.fromarray((img_fusion * 255).astype(np.uint8))
        # plt.figure()
        # plt.title('fuse')
        # plt.imshow(img_fusion, cmap='gray')
        # plt.show()
        final_fusion[i, :, :] = img_fusion

    # Perform Global MaxMinNormalization**
    # print("==================================before normalization===========================")
    # print(final_fusion)
    # print(np.min(final_fusion))
    # print(np.max(final_fusion))
    final_fusion = (final_fusion - np.min(final_fusion)) / (np.max(final_fusion) - np.min(final_fusion))
    final_fusion = np.clip(final_fusion * 255, 0, 255)
    # print("==================================after normalization===========================")
    # print(final_fusion)

    # for i in range(len(final_fusion)):
    #     plt.figure()
    #     plt.title('fuse')
    #     plt.imshow(final_fusion[i], cmap='gray')
    #     plt.show()
    return final_fusion


def block_to_img(block_img, m, n):
    '''
    Enter the fused block and restore it to the original image size.
    '''
    block_size = block_img.shape[2]
    m_block = int(np.ceil(m / block_size))
    n_block = int(np.ceil(n / block_size))
    fused_full_img_wpad = np.zeros((m_block * 256, n_block * 256), dtype=float)  # Image size after padding
    for i in range(0, m_block):
        for j in range(0, n_block):
            fused_full_img_wpad[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size] = block_img[i * n_block + j, :, :]
        fused_full_img = fused_full_img_wpad[:m, :n]  # image with original size
    return fused_full_img


def block_fusion(img1, img2, block_size=256):
    '''
    Input img1, img2, slice block according to block_size and fuse, output result
    '''
    # blocks_img大小[blocks_num, block_size, block_size, 3]
    blocks_img1 = get_block(img1, block_size=block_size)
    blocks_img2 = get_block(img2, block_size=block_size)
    # print('img1', blocks_img1.shape)
    # print('img2', blocks_img2.shape)

    # fusion
    fused_block_img1 = fuse(blocks_img1, blocks_img2)

    # block restore to orginal size
    fused_img = block_to_img(fused_block_img1, img1.shape[0], img1.shape[1])
    # Perform Global MaxMinNormalization**
    # fused_img = (fused_img - np.min(fused_img)) / (np.max(fused_img) - np.min(fused_img))
    # fused_img = np.clip(fused_img * 255, 0, 255)
    # plt.figure()
    # plt.title('fused_img')
    # plt.imshow(fused_img, cmap='gray')
    # plt.show()
    return fused_img


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def load_img(img_path):
    img = Image.open(img_path)
    img = img.convert('L')
    return _tensor(img).unsqueeze(0)


def load_img_cv(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    return img


def read_image(path):
    I = np.array(Image.open(path))
    return I


def fusion(x1, x2, model):
    with torch.no_grad():
        # start = time.time()
        feature1 = model.encoder(x1.float())
        feature2 = model.encoder(x2.float())

        if args.method==True:
            # 计算s
            from culw import CustomModel
            vgg = vgg19(pretrained=True).features
            vgg = vgg.to(device)
            for param in vgg.parameters():
                param.requires_grad = False
            culs = CustomModel(vgg,device)
            img_ue = torch.cat((x1, x1, x1), 1)
            img_oe = torch.cat((x2, x2, x2), 1)
            s = culs(img_ue,img_oe)

            fusion_feats = (s[0]*feature1 + s[1]*feature2) / 2
        else:
            fusion_feats = (feature1 + feature2) / 2

        # out = model.decoder(fusion_feats).squeeze(0).detach().cpu()
        # out = model.decoder(fusion_feats).squeeze(0).detach().cpu().numpy()
        out = model.decoder(fusion_feats).squeeze(0).detach().cpu()
        # time_used = time.time() - start
        # print("fusion time：", time_used, " used")
        return out





def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def load_img(img_path):
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.open(img_path, mode='L')
    return _tensor(img).unsqueeze(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='model save and load')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--ue_path', type=str, default='dataset/MEFB-main/dataset/MEFB-main/under-exposed/')
parser.add_argument('--oe_path', type=str, default='dataset/MEFB-main/dataset/MEFB-main/over-exposed/')
parser.add_argument('--model_path', type=str,
                    default='train_result/5.27-4-18.pth')
parser.add_argument('--model_path2', type=str,
                    default="train_result/5.11-3.pth")
parser.add_argument('--save_path', type=str, default='./MEFB_clif_/')
parser.add_argument('--resize', type=bool, default=False)
parser.add_argument('--method', type=bool, default=True)

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

        # img_ue = _tensor(img_ue).squeeze(0).to(device)
        # img_oe = _tensor(img_oe).squeeze(0).to(device)

        fusion_img = block_fusion(img_ue,img_oe)

        # fusion_img = normalization(fusion_img)
        # Perform Global MaxMinNormalization**
        # fusion_img = (fusion_img - np.min(fusion_img)) / (np.max(fusion_img) - np.min(fusion_img))
        # fusion_img = np.clip(fusion_img * 255, 0, 255)

        # print(fusion_img)
        # fusion_img = fusion_img.squeeze()

        # cv2.imwrite(args.save_path+name,fusion_img*255)

        # fusion_img_array = Image.fromarray((fusion_img * 255).astype(np.uint8))
        fusion_img_array = Image.fromarray((fusion_img).astype(np.uint8))
        fusion_img_array.save(args.save_path + name)

