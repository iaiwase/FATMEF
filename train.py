# -*- coding: utf-8 -*-
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import numpy as np

from FATMEF import CLMEFNet

from crLoss import CRLoss
from crinfoloss import CRInfoLoss
from LossW import LossW

from torchvision import transforms
from dataloader_clif_gray import Fusionset

from matplotlib import pyplot as plt
from ssim import SSIM, TV_Loss
from mefssim import MEFSSIM
import torch
from torchvision.models import vgg16
from torchvision.models import vgg19
import time
import argparse
import copy
from tensorboardX import SummaryWriter
# torch.set_printoptions(profile="full")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NWORKERS = 8

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='clif_try', help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--root', type=str, default='./COCO_color/train2017', help=' data path')
parser.add_argument('--save_path', type=str, default='./train_result_20mse_gray_1lr_0.01swap', help='model save path')
parser.add_argument('--model_type', type=str, default='CNN', help='model type')
parser.add_argument('--CR', type=bool, default=True, help='to choose a mini dataset')
parser.add_argument('--miniset', type=bool, default=True, help='to choose a mini dataset')
parser.add_argument('--minirate', type=float, default=0.05, help='to detemine the size of a mini dataset')
parser.add_argument('--seed', type=int, default=3, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--img_size', type=int, default=256, help='img size')

parser.add_argument('--samplelist', type=str, default='./extreme_MEF_pool_gray_1/', help='model samplelist')
parser.add_argument('--labellist', type=str, default='./extreme_MEF_pool_gray_1/', help='model samplelist')

parser.add_argument('--perloss', type=bool, default=True, help='using perloss')
parser.add_argument('--ssimloss', type=bool, default=True, help='using perloss')
parser.add_argument('--tvloss', type=bool, default=True, help='using perloss')
parser.add_argument('--crloss', type=bool, default=False, help='using perloss')
parser.add_argument('--localloss', type=bool, default=False, help='using perloss')
parser.add_argument('--InfoNCELoss', type=bool, default=False, help='using perloss')

parser.add_argument('--w_crloss', type=float, default=0.08, help='using perloss')
parser.add_argument('--w_mseloss', type=float, default=1, help='using perloss')
parser.add_argument('--w_l1loss', type=float, default=1, help='using perloss')
parser.add_argument('--w_ssimloss', type=float, default=1, help='using perloss')
parser.add_argument('--w_tvloss', type=float, default=20, help='using perloss')
parser.add_argument('--w_localloss', type=float, default=0.1, help='using perloss')
parser.add_argument('--w_infonceloss', type=float, default=20, help='using perloss')

parser.add_argument('--epoch', type=int, default=35, help='training epoch')
parser.add_argument('--batch_size', type=int, default=28, help='batchsize')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--summary_name', type=str, default='clif_try_', help='Name of the summmary')

args = parser.parse_args()
writer = SummaryWriter(comment=args.summary_name)
# ==================
# init
# ==================
# io = log.IOStream(args)
print(str(args))
toPIL = transforms.ToPILImage()
np.random.seed(1)  # to get the same images leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    print('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
          str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    print('Using CPU')


# ==================
# Read Data
# ==================

def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def train(model, train_loader, val_loader, optimizer, criterion, args):
    # Training
    loss_val = []

    mkdir(args.save_path)

    for epoch in tqdm(range(args.epoch)):

        all_loss = 0.
        all_mse_loss = 0.
        all_ssim_loss = 0.
        all_tv_loss = 0.
        all_cr_loss = 0.
        all_l1_loss = 0.
        all_loss_local_cr = 0
        all_loss_infoNCE = 0

        model.train()
        for index, image in enumerate(tqdm(train_loader)):

            img_orig = image[0].to(device)  # shape:[B,1,256,256] I0
            img_trans = image[1].to(device)  # shape:[B,1,256,256] It
            img_deip = image[2].to(device) # DEIP
            img_label = image[3].to(device) # Label
            optimizer.zero_grad()
            img_recon = model(img_trans.float())  # If

            loss = criterion[0](img_recon, img_orig)
            loss_mse = loss
            if args.perloss:
                loss_cr, loss_local_cr = criterion[1](torch.cat((img_recon, img_recon, img_recon), 1),
                                                                    torch.cat((img_orig, img_orig, img_orig), 1),
                                                                    torch.cat((img_deip, img_deip, img_deip), 1))

                if args.ssimloss==True:
                    loss_ssim = 0
                    batchsize = img_recon.shape[0]
                    for i in range(0, batchsize):
                        loss_ssim = (1 - criterion[2](img_recon[i].unsqueeze(0), img_orig[i].unsqueeze(0)))
                        if (np.isnan(loss_ssim.item()) == 0):
                            loss_ssim += loss_ssim.item()
                else:
                    loss_ssim = torch.tensor(0.0)
                loss_tv = criterion[3](img_recon, img_orig)
                loss_l1 = criterion[4](img_recon, img_orig)
                loss_infoNCE = criterion[6](torch.cat((img_recon, img_recon, img_recon), 1),
                                            torch.cat((img_label, img_label, img_label), 1),
                                            torch.cat((img_trans, img_trans, img_trans), 1))
                loss = args.w_mseloss * loss_mse + args.w_crloss * loss_cr + args.w_ssimloss * loss_ssim \
                       + args.w_tvloss * loss_tv + args.w_l1loss * loss_l1 + args.w_localloss * loss_local_cr + args.w_infonceloss * loss_infoNCE

            loss.backward()
            optimizer.step()

            all_loss += loss.item()
            all_mse_loss += loss_mse.item()
            all_ssim_loss += loss_ssim.item()
            all_tv_loss += loss_tv.item()
            all_cr_loss += loss_cr.item()
            all_l1_loss += loss_l1.item()
            all_loss_local_cr += loss_local_cr.item()
            all_loss_infoNCE += loss_infoNCE.item()
        
        print("all_cr_loss:",all_cr_loss/(len(train_loader)))
        print("all_tv_loss:",all_tv_loss/(len(train_loader)))
        print("ssimloss:",all_ssim_loss/(len(train_loader)))
        print('Epoch:[%d/%d]-----Train------ LOSS:%.4f' % (epoch, args.epoch, all_loss / (len(train_loader))))
        
        writer.add_scalar('Train/loss', all_loss / (len(train_loader)), epoch)
        writer.add_scalar('Train/mse_loss', all_mse_loss / (len(train_loader)), epoch)
        writer.add_scalar('Train/ssim_loss', all_ssim_loss / (len(train_loader)), epoch)
        writer.add_scalar('Train/tv_loss', all_tv_loss / (len(train_loader)), epoch)
        writer.add_scalar('Train/cr_loss', all_cr_loss / (len(train_loader)), epoch)
        writer.add_scalar('Train/l1_loss', all_l1_loss / (len(train_loader)), epoch)
        writer.add_scalar('Train/local_cr_loss', all_loss_local_cr / (len(train_loader)), epoch)
        writer.add_scalar('Train/infoNCE_loss', all_loss_infoNCE / (len(train_loader)), epoch)
        writer.add_scalar('Train/memory', torch.cuda.memory_allocated() / 1024 / 1024, epoch)


        scheduler.step()

        model.eval()
        with torch.no_grad():

            all_loss = 0.
            all_mse_loss = 0.
            all_ssim_loss = 0.
            all_tv_loss = 0.
            all_cr_loss = 0.
            all_l1_loss = 0.
            all_loss_local_cr = 0
            all_loss_infoNCE = 0

            for index, image in enumerate(tqdm(val_loader)):
                img_orig = image[0].to(device)  # shape:[B,1,256,256] I0
                img_trans = image[1].to(device)  # shape:[B,1,256,256] It
                img_deip = image[2].to(device)  # DEIP
                img_label = image[3].to(device)  # Label

                img_recon = model(img_orig.float())

                loss = criterion[0](img_recon, img_orig)
                loss_mse = loss
                if args.perloss:
                    loss_cr, loss_local_cr = criterion[1](torch.cat((img_recon, img_recon, img_recon), 1),
                                                                        torch.cat((img_orig, img_orig, img_orig), 1),
                                                                        torch.cat((img_deip, img_deip, img_deip), 1))

                    if args.ssimloss==True:
                        loss_ssim = 0
                        batchsize = img_recon.shape[0]
                        for i in range(0, batchsize):
                            loss_ssim = (1 - criterion[2](img_recon[i].unsqueeze(0), img_orig[i].unsqueeze(0)))
                            if (np.isnan(loss_ssim.item()) == 0):
                                loss_ssim += loss_ssim.item()
                    else:
                        loss_ssim = torch.tensor(0.0)
                    loss_tv = criterion[3](img_recon, img_orig)
                    loss_l1 = criterion[4](img_recon, img_orig)
                    loss_infoNCE = criterion[6](torch.cat((img_recon, img_recon, img_recon), 1),torch.cat((img_label, img_label, img_label), 1),torch.cat((img_trans, img_trans, img_trans), 1))
                    
                    
                    loss = args.w_mseloss * loss_mse + args.w_crloss * loss_cr + args.w_ssimloss * loss_ssim \
                       + args.w_tvloss * loss_tv + args.w_l1loss * loss_l1 + args.w_localloss * loss_local_cr + args.w_infonceloss * loss_infoNCE
                all_loss += loss.item()
                all_mse_loss += loss_mse.item()
                all_ssim_loss += loss_ssim.item()
                all_tv_loss += loss_tv.item()
                all_cr_loss += loss_cr.item()
                all_l1_loss += loss_l1.item()
                all_loss_local_cr += loss_local_cr.item()
                all_loss_infoNCE += loss_infoNCE.item()
        
        print("all_cr_loss:",all_cr_loss/(len(val_loader)))
        print("all_tv_loss:",all_tv_loss/(len(val_loader)))
        print("ssimloss:",all_ssim_loss/(len(val_loader)))
        print('Epoch:[%d/%d]---Validation--- LOSS:%.4f' % (epoch, args.epoch, all_loss / (len(val_loader))))
        
        writer.add_scalar('Val/loss', all_loss / (len(val_loader)), epoch)
        writer.add_scalar('Val/mse_loss', all_mse_loss / (len(val_loader)), epoch)
        writer.add_scalar('Val/ssim_loss', all_ssim_loss / (len(val_loader)), epoch)
        writer.add_scalar('Val/tv_loss', all_tv_loss / (len(val_loader)), epoch)
        writer.add_scalar('Val/cr_loss', all_cr_loss / (len(val_loader)), epoch)
        writer.add_scalar('Val/l1_loss', all_l1_loss / (len(val_loader)), epoch)
        writer.add_scalar('Val/local_cr_loss', all_loss_local_cr / (len(train_loader)), epoch)
        writer.add_scalar('Val/infoNCE_loss', all_loss_infoNCE / (len(train_loader)), epoch)

        loss_val.append(all_loss / (len(val_loader)))

        # save model every epoch
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
        }
        torch.save(state, os.path.join(args.save_path, args.summary_name + str(epoch) + '.pth'))

    ### save best model###
    minloss_index = loss_val.index(min(loss_val))
    print("The min loss in validation is obtained in %d epoch" % (minloss_index))
    print("The training process has finished! Take a break! ")


if __name__ == "__main__":
    train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(args.img_size),
                                                         torchvision.transforms.RandomCrop(args.img_size),
                                                         torchvision.transforms.RandomHorizontalFlip()])

    dataset = Fusionset(args, args.root, transform=train_augmentation, gray=True, partition='train', CR=args.CR)
    # Creating data indices for training and validation splits:
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)  # sampler will assign the whole data accordinig to batchsize.
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, num_workers=NWORKERS, batch_size=args.batch_size,
                              sampler=train_sampler, drop_last=True)  # len(train_loader)*batchsize = len(train_sampler)
    val_loader = DataLoader(dataset, num_workers=NWORKERS, batch_size=args.batch_size,
                            sampler=valid_sampler)
    model = CLMEFNet().to(device)

    model_dict = model.state_dict()
    # 加载.pth文件
    model_state_dict = torch.load('checkpoint/swin_base_patch4_window7_224_22k.pth', map_location=torch.device('cpu'))
    for k, v in model_state_dict.items():
        model_state_dict = v
    state_dict = {"encoder.transformer." + k: v for k, v in model_state_dict.items() if
                  "encoder.transformer." + k in model_dict.keys()}
    #     for name, para in model.named_parameters():
    #         if name in state_dict.keys():
    #             print(name,":requires_grad is False")
    #             para.requires_grad_(False)
    model.load_state_dict(state_dict, strict=False)

    criterion = []
    criterion.append(nn.MSELoss().to(device))

    # criterion[1]
#     vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg19(pretrained=True).features
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    #     criterion.append(LossNetwork(vgg_model).to(device))
    criterion.append(CRLoss(vgg_model, args).to(device))

#     criterion.append(SSIM().to(device))
    criterion.append(MEFSSIM().to(device))# 2
    criterion.append(TV_Loss().to(device))# 3
    criterion.append(nn.L1Loss().to(device))# 4
    criterion.append(LossW(vgg_model).to(device))# 5
    criterion.append(CRInfoLoss(vgg_model, args).to(device))# 6


    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    scheduler = CosineAnnealingLR(optimizer, args.epoch)
    optimizer.zero_grad()
    # Handle multi-gpu
    if (device.type == 'cuda') and len(args.gpus) > 1:
        model = nn.DataParallel(model, args.gpus)

    print('============ Training Begins ===============')
    train(model, train_loader, val_loader, optimizer, criterion, args)
