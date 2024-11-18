import segmentation_models_pytorch_3d as smp
import torch

device = torch.device("cuda:0")

model = smp.Unet(
    encoder_name="resnet50",    
    in_channels=4,             
    classes=2,                 
    strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1))  
).to(device)

