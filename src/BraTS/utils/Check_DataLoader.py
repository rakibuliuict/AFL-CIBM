import argparse
from train_file.DataLoader import prepare
import matplotlib.pyplot as plt


def main(data_dir):
    
    train_loader, test_loader = prepare(data_dir)

    
    val_data_example = test_loader.dataset[5]  
    image = val_data_example['image']
    label = val_data_example['label']

    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")

    
    plt.figure("Image Channels", (24, 4))
    for i in range(image.shape[0]):  
        plt.subplot(1, 4, i + 1)
        plt.title(f"Image channel {i}")
        plt.imshow(image[i, :, :, 60].detach().cpu(), cmap="gray")
    plt.show()

 
    plt.figure("Label Channels", (18, 4))
    for i in range(label.shape[0]):  
        plt.subplot(1, 3, i + 1)
        plt.title(f"Label channel {i}")
        plt.imshow(label[i, :, :, 60].detach().cpu())
    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Visualize dataset images and labels")

    
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")

    
    args = parser.parse_args()

    
    main(data_dir=args.data_dir)
