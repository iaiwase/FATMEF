import torch
import random
import torch.nn.functional as F

class CRInfoLoss(torch.nn.Module):
    def __init__(self, vgg_model,args):
        super(CRInfoLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.Info = args.InfoNCELoss 
        self.layer_name_mapping_info = {
            '29': "4"
        } 

    
    def output_features_info(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping_info:
                output[self.layer_name_mapping_info[name]] = x
        return list(output.values())

    def forward(self, fuse, positive,negative, patchSize=32, pathNum=4, t=0.01):
        if self.Info==True:
            # InfoNCELoss

            p_f_loss = 0
            n_f_loss = 0
            infonce_loss = 0
            p_f_all_loss = []
            n_f_all_loss = []
            infonce_all_loss = []

            positive_features = self.output_features_info(positive)
            negative_features = self.output_features_info(negative)
            fuse_features = self.output_features_info(fuse)
            for positive_feature,negative_feature,fuse_feature in zip(positive_features,negative_features,fuse_features):
                #InfoNCELoss
                m = positive_feature.size(3)
                n = negative_feature.size(2)
                s_pf = torch.sum(torch.div(torch.mul(positive_feature, fuse_feature), torch.norm(positive_feature, p=2)*torch.norm(fuse_feature, p=2)))/m/n
                s_nf = torch.sum(torch.div(torch.mul(negative_feature, fuse_feature), torch.norm(negative_feature, p=2)*torch.norm(fuse_feature, p=2)))/m/n
                infonce_loss+=(-torch.log(torch.exp(s_pf/t)/torch.add(torch.exp(s_pf/t),torch.exp(s_nf/t))))
                
            loss_infoNCE = infonce_loss
        else:
            loss_infoNCE=torch.tensor(0.0)
        
        return loss_infoNCE