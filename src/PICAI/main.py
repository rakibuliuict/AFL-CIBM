
import argparse
from training_setup.model import *
from training_setup.Dataloader import *
from training_setup.train import *

def main(data_dir, model_dir, epochs):
    data_in = prepare(data_dir, cache=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-5)

    
    train(model, data_in, optimizer, epochs, model_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a 3D segmentation model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    args = parser.parse_args()

    main(args.data_dir, args.model_dir, args.epochs)
