import torch
from monai.transforms import MapTransform

class ConvertToBinaryClassBasedOnBratsClassesd(MapTransform):
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            binary_label = torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 3)
            d[key] = binary_label.float().unsqueeze(0)  
        return d
