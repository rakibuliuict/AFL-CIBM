import os
import torch
import numpy as np
import nibabel as nib
import argparse
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import Activations
import segmentation_models_pytorch_3d as smp
from training_setup.Dataloader import prepare  

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

  
    train_loader, test_loader = prepare(in_dir=in_dir, cache=False)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = smp.Unet(
        encoder_name="resnet50",        
        in_channels=3,                  
        strides=((2, 2, 2), (4, 2, 1), (2, 2, 2), (2, 2, 1), (1, 2, 3)),
        classes=2
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth"), map_location=device))
    model.eval()

   
    os.makedirs(output_folder, exist_ok=True)


    sw_batch_size = 1
    roi_size = (160, 128, 24)

    with torch.no_grad():
        for idx, test_patient in enumerate(test_loader):
   
            patient_id = os.path.basename(test_patient['t2w'].meta['filename_or_obj'][0]).replace('.nii.gz', '')
            print(f"Processing patient {patient_id} ({idx + 1}/{len(test_loader)})")

            label = test_patient["seg"]
            t_volume = test_patient['img']

         
            test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)

  
            sigmoid_activation = Activations(sigmoid=True)
            test_outputs = sigmoid_activation(test_outputs)
            test_outputs = test_outputs > 0.50
            test_outputs = test_outputs.int()

         
            output_data = test_outputs.cpu().numpy()[0, 1].astype(np.int16)

          
            input_image = nib.load(test_patient['t2w'].meta['filename_or_obj'][0])
            affine = input_image.affine

           
            nifti_image = nib.Nifti1Image(output_data, affine)
            output_file = os.path.join(output_folder, f"{patient_id}-pred.nii.gz")
            nib.save(nifti_image, output_file)
            print(f"Saved prediction for {patient_id} at {output_file}")

if __name__ == "__main__":
    main()
