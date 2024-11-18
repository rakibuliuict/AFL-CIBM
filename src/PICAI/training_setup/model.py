from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from training_setup.neural_networks.Architectures import U_Net3D, VNet, SegResNet,Incre_MRRN_v2_3d
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from monai.networks.nets import UNet,AttentionUnet,RegUNet
from monai.networks.layers import Norm

import tqdm
import segmentation_models_pytorch_3d as smp
import torch


device = torch.device("cuda:0")

model = smp.Unet(
    encoder_name="resnet50",        
    in_channels=3,                  
    strides=((2, 2, 2), (4, 2, 1), (2, 2, 2), (2, 2, 1), (1, 2, 3)),
    classes=2, 
).to(device)

##model_raw_code------------------------------------------------------------------------------------------------------------

