import torch
import random
import torch.nn.functional as F

class CRLoss(torch.nn.Module):
    def __init__(self, vgg_model,args):
        super(CRLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.cr = args.crloss
        # vgg16
        self.layer_name_mapping_cr = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features_cr(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping_cr:
                output[self.layer_name_mapping_cr[name]] = x
        return list(output.values())
    
    def output_features_local(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping_local:
                output[self.layer_name_mapping_local[name]] = x
        return list(output.values())
    
    def output_features_info(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping_info:
                output[self.layer_name_mapping_info[name]] = x
        return list(output.values())

    def forward(self, fuse, positive,negative):
        if self.cr==True:
            # CRLoss
            p_f_loss = []
            n_f_loss = []
            positive_features = self.output_features_cr(positive)
            negative_features = self.output_features_cr(negative)
            fuse_features = self.output_features_cr(fuse)
            for positive_feature,negative_feature, fuse_feature in zip(positive_features,negative_features, fuse_features):
                p_f_loss.append(F.mse_loss(positive_feature, fuse_feature))
                n_f_loss.append(F.mse_loss(negative_feature, fuse_feature))
            loss_cr = torch.max((sum(p_f_loss) - sum(n_f_loss) + 0.2),torch.tensor(0.0))
#             loss_cr = (sum(p_f_loss)/len(p_f_loss))/(sum(n_f_loss)/len(n_f_loss))
#             loss_cr = sum(p_f_loss)/len(p_f_loss)
        else:
            loss_cr = torch.tensor(0.0)

        return loss_cr
