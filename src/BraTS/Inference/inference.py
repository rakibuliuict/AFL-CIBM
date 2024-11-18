import os
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
from monai.transforms import Activations
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from train_file.DataLoader import prepare
import segmentation_models_pytorch_3d as smp
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for segmentation model")
    parser.add_argument('--in_dir', type=str, required=True, help="Path to the input data directory")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the trained model directory")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output folder for saving predictions")
    return parser.parse_args()


def main():

    args = parse_args()


    in_dir = args.in_dir
    model_dir = args.model_dir
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

  
    _, test_loader = prepare(in_dir=in_dir, cache=False)

   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  
    model = smp.Unet(
        encoder_name="resnet50",    
        in_channels=4,             
        classes=2,                 
        strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1))  
    ).to(device)

    model_path = os.path.join(model_dir, "best_metric_model.pth")
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

   
    sw_batch_size = 1
    roi_size = (224, 224, 144)  

 
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Inference Progress")):
            patient_id = batch['patient_id'][0]
            print(f"Processing patient: {patient_id}")

            
            outputs = sliding_window_inference(
                inputs=batch['image'].to(device), 
                roi_size=roi_size, 
                sw_batch_size=sw_batch_size, 
                predictor=model
            )

            
            sigmoid_activation = Activations(sigmoid=True)
            predicted = sigmoid_activation(outputs) > 0.5
            predicted = predicted.int()

            
            output_data = predicted.cpu().numpy()[0, 1].astype(np.int16)
            input_image = nib.load(batch['image'].meta['filename_or_obj'][0])
            nifti_image = nib.Nifti1Image(output_data, input_image.affine)
            output_file = os.path.join(output_folder, f"{patient_id}-pred.nii.gz")
            nib.save(nifti_image, output_file)
            print(f"Saved prediction to: {output_file}")

if __name__ == "__main__":
    main()
