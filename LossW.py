import torch
import torch.nn.functional as F


# --- Perceptual loss network  --- #
class LossW(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossW, self).__init__()
        self.vgg = vgg_model

    def forward(self, img_deip,img_trans,img_orig,img_label,t=0.01):
        img_deip = self.vgg(img_deip)
        img_trans = self.vgg(img_trans)
        img_orig = self.vgg(img_orig)
        img_label = self.vgg(img_label)
        m = img_orig.size(3)
        n = img_orig.size(2)
        w_I0_It = torch.sum(torch.div(torch.mul(img_orig, img_trans),
                                        torch.norm(img_orig, p=2) * torch.norm(img_trans, p=2)))
        w_I0_It = max(w_I0_It.item()-0.8,0)/0.2/2+0.5
        return max(1-w_I0_It,0)
