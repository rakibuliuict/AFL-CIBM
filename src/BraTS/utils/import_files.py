import os
from glob import glob
import shutil
from tqdm import tqdm
# import dicom2nifti
import numpy as np
import nibabel as nib
from monai.transforms import (
    # AddChanneld,
    EnsureChannelFirstd,
    MapTransform,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    ToTensord,
    ScaleIntensityd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    EnsureTyped,
    Flipd,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ConcatItemsd,
    ScaleIntensityd,
    AdjustContrastd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    # LoadNiftid,
    # AddChanneld,

)
from monai.data import DataLoader, Dataset, CacheDataset,decollate_batch
from monai.utils import set_determinism
from monai.utils import first
import matplotlib.pyplot as plt
import torch
import monai
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
from monai.utils import first
import matplotlib.pyplot as plt
import torch
import monai

import os
from glob import glob
from monai.data import Dataset, CacheDataset, DataLoader
from monai.transforms import Compose