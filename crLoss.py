import torch
import random
import torch.nn.functional as F

class CRLoss(torch.nn.Module):
    def __init__(self, vgg_model,args):
        super(CRLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.cr = args.crloss
        self.local = args.localloss
        self.Info = args.InfoNCELoss
        # vgg16
        self.layer_name_mapping_cr = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }
        self.layer_name_mapping_local = {
            '1': "1",
            '4': "2",
            '8': "3",
            '22': "4"
        }   
        self.layer_name_mapping_info = {
            '3': "1",
            '29': "4"
        } 
        # vgg19
#         self.layer_name_mapping = {
#             '3': "relu1_2",
#             '8': "relu2_2",
#             '17': "relu3_3"
#         }
#         self.layer_name_mapping = {
#             '3': "relu1_2",
#             '8': "relu2_2",
#             '17': "relu3_3"
#         }      

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

    def forward(self, fuse, positive,negative, patchSize=32, pathNum=4, t=0.01):
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
        if self.local==True:
            # LocalLoss
            w = positive.size(3)
            h = positive.size(2)

            p_f_loss = 0
            n_f_loss = 0
            infonce_loss = 0
            k=0
            p_f_all_loss = []
            n_f_all_loss = []
            infonce_all_loss = []
            for i in range(pathNum):
                p_f_loss=0
                n_f_loss=0
                infonce_loss=0
                k=0
                w_offset_1 = random.randint(
                    0, max(0, w - patchSize - 1))
                h_offset_1 = random.randint(
                    0, max(0, h - patchSize - 1))
                positive_patch=positive[:, :, h_offset_1:h_offset_1 + patchSize,
                               w_offset_1:w_offset_1 + patchSize]
                negative_patch=negative[:, :, h_offset_1:h_offset_1 + patchSize,
                               w_offset_1:w_offset_1 + patchSize]
                fuse_patch=fuse[:, :, h_offset_1:h_offset_1 + patchSize,
                               w_offset_1:w_offset_1 + patchSize]
                positive_patch_features = self.output_features_local(positive_patch)
                negative_patch_features = self.output_features_local(negative_patch)
                fuse_patch_features = self.output_features_local(fuse_patch)
                for positive_patch_feature,negative_patch_feature,fuse_patch_feature in zip(positive_patch_features,negative_patch_features,fuse_patch_features):
                    k+=1
                    #LocalLoss
                    p_f_loss+=(F.mse_loss(positive_patch_feature, fuse_patch_feature))
                    n_f_loss+=(F.mse_loss(negative_patch_feature, fuse_patch_feature))
                p_f_all_loss.append((p_f_loss) / k)
                n_f_all_loss.append((n_f_loss) / k)
            loss_local_positive = sum(p_f_all_loss)
            loss_local_negative = sum(n_f_all_loss)
            loss_local_cr = loss_local_positive / loss_local_negative
        else:
#             print("loss_local_cr=0")
            loss_local_cr=torch.tensor(0.0)
        
        return loss_cr,loss_local_cr