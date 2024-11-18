import argparse
import torch
from train_file.models import *
from train_file.train import *
from train_file.DataLoader import *


def main(data_dir, model_dir, num_epochs=200):

    data_in = prepare(data_dir, cache=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-5)
    
    train(model, data_in, optimizer, num_epochs, model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified data and model directory")

    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to save the trained model")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs to train the model")

    args = parser.parse_args()

    main(data_dir=args.data_dir, model_dir=args.model_dir, num_epochs=args.num_epochs)
