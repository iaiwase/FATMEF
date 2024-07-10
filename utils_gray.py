import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch import fft

def normalization(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / _range


# for Fourier_Domain_Swapping
def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:, :, :, :, 0] ** 2 + fft_im[:, :, :, :, 1] ** 2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
    return fft_amp, fft_pha


def low_freq_mutate_torch(amp_src, amp_trg, L=0.1):
    a_src = torch.fft.fftshift(amp_src, dim=(1,2))
    a_trg = torch.fft.fftshift(amp_trg, dim=(1,2))

    h, w = a_src.shape[-2:]
    b = int(torch.floor(torch.min(torch.tensor([h, w])).float() * L))
    c_h = torch.floor(torch.tensor(h / 2.0)).to(torch.int)
    c_w = torch.floor(torch.tensor(w / 2.0)).to(torch.int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[..., h1:h2, w1:w2] = a_trg[..., h1:h2, w1:w2]
    a_src = torch.fft.ifftshift(a_src, dim=(1,2))
    return a_src


def FDA_source_to_target_np(src_img, trg_img, L=0.01):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_torch = src_img
    trg_img_torch = trg_img

    # get fft of both source and target
    fft_src_torch = fft.fft2(src_img_torch, dim=(1,2)) # [1,256,256]
    fft_trg_torch = fft.fft2(trg_img_torch, dim=(1,2)) # [1,256,256]

    # extract amplitude and phase of both ffts
    amp_src, pha_src = torch.abs(fft_src_torch), torch.angle(fft_src_torch)
    amp_trg, pha_trg = torch.abs(fft_trg_torch), torch.angle(fft_trg_torch)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_torch(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * torch.exp(1j * pha_src)
    
    # get the mutated image
    src_in_trg = fft.ifft2(fft_src_, dim=(1,2))
    src_in_trg = torch.real(src_in_trg)
    
    return src_in_trg
