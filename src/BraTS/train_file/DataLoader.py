import os
from glob import glob
from monai.data import Dataset, CacheDataset, DataLoader
from monai.utils import set_determinism
from augmentations.test_augment import get_test_transforms
from augmentations.train_augment import get_train_transforms
from typing import Tuple, List, Dict

def prepare(in_dir: str, cache: bool = False) -> Tuple[DataLoader, DataLoader]:
    set_determinism(seed=0)
    

    path_train_volumes_t2w = sorted(glob(os.path.join(in_dir, "image_Tr", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "label_Tr", "*.nii.gz")))
    path_test_volumes_t2w = sorted(glob(os.path.join(in_dir, "image_Ts", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "label_Ts", "*.nii.gz")))

    def extract_patient_id(filename: str) -> str:

        parts = filename.split("_")
        if len(parts) == 2 and parts[1].endswith(".nii.gz"):
            return parts[1].split(".")[0]
        return None


    train_files = [{"image": t2w_image, "label": label_name}
                   for t2w_image, label_name in zip(path_train_volumes_t2w, path_train_segmentation)]

    test_files = [{"image": img, "label": lbl, "patient_id": extract_patient_id(os.path.basename(img))}
                  for img, lbl in zip(path_test_volumes_t2w, path_test_segmentation)]


    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()


    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        test_ds = Dataset(data=test_files, transform=test_transforms)


    train_loader = DataLoader(train_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)

    return train_loader, test_loader
