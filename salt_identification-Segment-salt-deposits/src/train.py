from torch.autograd import Variable
from tqdm import tqdm, tqdm_notebook
import torch
import os
import argparse
import numpy as np
from torch.utils import data
from model import get_model
from load_data import define_paths
from load_data import Dataset


def main():
    parser = argparse.ArgumentParser(
        description='A script for training the data')
    parser.add_argument('--image_path',
                        required=True,
                        type=str,
                        help='target segmented binary image.')
    parser.add_argument('--dir',
                        required=False,
                        default="input/",
                        type=str,
                        help='train.csv path')
    parser.add_argument('--lr',
                        required=False,
                        default=0.001,
                        type=float,
                        help='learning rate')
    parser.add_argument('--optm',
                        required=False,
                        default=torch.optim.Adam,
                        type=str,
                        help='optimizer')
    args = parser.parse_args()

    checkpoint_path = "model_checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    
    model = get_model()  # get_model() <- model.py
    optimizer = args.optm(model.parameters(), lr=args.lr)
    save_checkpoint(checkpoint_path, model, optimizer)
    load_checkpoint(checkpoint_path, model, optimizer)
    train(model, args.dir)


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def train(get_model, directory):
    train_path, file_list = define_paths(directory)  # define_paths(): func <- load_data.py
    file_list_val = file_list[::10]
    file_list_train = [f for f in file_list if f not in file_list_val]
    dataset = Dataset(train_path, file_list_train)  # Dataset: class <- load_data.py
    dataset_val = Dataset(train_path, file_list_val)

    model = get_model
    epoch = 13
    learning_rate = 1e-4
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for e in range(epoch):
        train_loss = []
        for image, mask in tqdm_notebook(
                data.DataLoader(dataset, batch_size=30, shuffle=True)):
            image = image.type(torch.FloatTensor).cuda()
            y_pred = model(Variable(image))
            loss = loss_fn(y_pred, Variable(mask.cuda()))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss.append(loss.data[0])

        val_loss = []
        for image, mask in data.DataLoader(dataset_val,
                                           batch_size=50,
                                           shuffle=False):
            image = image.cuda()
            y_pred = model(Variable(image))

            loss = loss_fn(y_pred, Variable(mask.cuda()))
            val_loss.append(loss.data[0])

        print("Epoch: %d, Train: %.3f, Val: %.3f" %
              (e, np.mean(train_loss), np.mean(val_loss)))
    # save the final model
    save_checkpoint('tgs-%i.pth' % epoch, model, optimizer)


if __name__ == "__main__":
    main()