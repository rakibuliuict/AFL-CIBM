import argparse
from training_setup.Dataloader import prepare

def main(data_dir):

    train_loader, test_loader = prepare(data_dir)


    for batch_idx, batch in enumerate(train_loader):
        data = batch["img"]
        label = batch["seg"]
        print(f"Batch {batch_idx + 1} - Data shape: {data.shape}")
        print(f"Batch {batch_idx + 1} - Label shape: {label.shape}")

        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and process dataset for training.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Path to the dataset directory containing training and validation volumes."
    )
    args = parser.parse_args()
    main(args.data_dir)
