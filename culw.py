import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self,vgg,device):
        super(CustomModel, self).__init__()
        self.c = 1
        self.vgg1 = vgg
        self.vgg2 = vgg
        self.vgg_layers = vgg
        self.device = device
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "1",
            '29': "2"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def features_grad(self,features):
        kernel = torch.tensor([[1 / 8, 1 / 8, 1 / 8],
                               [1 / 8, -1, 1 / 8],
                               [1 / 8, 1 / 8, 1 / 8]], dtype=torch.float32).to(self.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        # print("kernel:\n",kernel.shape)
        # print("features:\n",features.shape)
        c = features.shape[1]
        c = int(c)
        fgs = []
        for i in range(c):
            fg = F.conv2d(features[:, i, :, :].unsqueeze(1), kernel, stride=1, padding=1)
            fgs.append(fg)
        fgs = torch.cat(fgs, dim=-1)
        return fgs

    def forward(self, S1_VGG_in, S2_VGG_in):
        S1_FEAS = self.output_features(S1_VGG_in)
        S2_FEAS = self.output_features(S2_VGG_in)

        ws1_list = []
        ws2_list = []
        for i in range(len(S1_FEAS)):
            m1 = torch.mean(torch.square(self.features_grad(S1_FEAS[i])), dim=[1, 2, 3])
            m2 = torch.mean(torch.square(self.features_grad(S2_FEAS[i])), dim=[1, 2, 3])
            ws1_list.append(m1.unsqueeze(-1))
            ws2_list.append(m2.unsqueeze(-1))

        ws1 = torch.cat(ws1_list, dim=-1)
        ws2 = torch.cat(ws2_list, dim=-1)

        # print("ws1:\n",ws1)
        # print("ws2:\n",ws2)
        # print("c:\n",self.c)
        s1 = torch.mean(ws1, dim=-1) / self.c
        s2 = torch.mean(ws2, dim=-1) / self.c
        # print(s1)
        # print(s2)
        s = F.softmax(torch.cat([s1, s2], dim=-1), dim=-1)

        return s
