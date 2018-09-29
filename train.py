import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import dataloader

from model.model import Network

import numpy as np
import random
import pickle as pkl
import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def collate_fn(data):
    data_x = []
    data_y = []
    for i, sample in enumerate(data):
        sample_x, sample_y = sample[0], sample[1]
        # normalizing
        sample_x = sample_x / 255.0
        sample_x = normalize(sample_x)
        # random-cropping
        random_x = random.randint(0, sample_y.shape[0]-224)
        random_y = random.randint(0, sample_y.shape[1]-224)
        sample_x = sample_x[:, random_x:random_x+224, random_y:random_y+224]
        sample_y = sample_y[random_x:random_x+224, random_y:random_y+224]
        data_x.append(sample_x)
        data_y.append(sample_y)
    data_x = torch.stack(data_x)
    data_y = torch.stack(data_y)
    return data_x, data_y


def main(params):

    print("Loading dataset ... ")

    with open(params['train_data_pkl'], 'rb') as f:
        train_data = pkl.load(f)
    with open(params['train_anno_pkl'], 'rb') as f:
        train_anno = pkl.load(f)
    """
    with open(params['val_data_pkl'], 'rb') as f:
        val_data = pkl.load(f)
    with open(params['val_anno_pkl'], 'rb') as f:
        val_anno = pkl.load(f)
    """

    # Train dataset and Train dataloader
    train_data = np.transpose(train_data, (0, 3, 1, 2))
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_data), torch.LongTensor(train_anno))

    train_loader = dataloader.DataLoader(
        train_dataset, params['batch_size'], shuffle=True, collate_fn=collate_fn)

    """
    # Validation dataset and Validation dataloader
    val_data = np.transpose(val_data, (0, 3, 1, 2))
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_data), torch.LongTensor(val_anno))
        val_loader = dataloader.DataLoader(
            val_dataset, params['batch_size'], collate_fn=collate_fn)
    """

    # the number of layers in each dense block
    n_layers_list = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]

    print("Constructing the network ... ")
    # Define the network
    densenet = Network(n_layers_list, 5).to(device)

    if os.path.isfile(params['model_from']):
        print("Starting from the saved model")
        densenet.load_state_dict(torch.load(params['model_from']))
    else:
        print("Couldn't find the saved model")
        print("Starting from the bottom")

    print("Training the model ...")
    # hyperparameter, optimizer, criterion
    learning_rate = params['lr']
    optimizer = torch.optim.RMSprop(
        densenet.parameters(), learning_rate, weight_decay=params['l2_reg'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(params['max_epoch']):
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)

            # forward-propagation
            pred = densenet(img)

            # flatten for all pixel
            pred = pred.view((-1, params['num_answers']))
            label = label.view((-1))

            # get loss
            loss = criterion(pred, label)

            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch: %d, Steps:[%d/%d], Loss: %.4f" %
                  (epoch, i, len(train_loader), loss.data))

        learning_rate *= 0.995
        optimizer = torch.optim.RMSprop(
            densenet.parameters(), learning_rate, weight_decay=params['l2_reg'])

        if (epoch+1) % 10 == 0:
            print("Saved the model")
            torch.save(densenet.state_dict(), params['model_save'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # dataset pickle
    parser.add_argument('--train_data_pkl', default='data/train.pickle')
    parser.add_argument('--train_anno_pkl', default='data/trainannot.pickle')
    parser.add_argument('--val_data_pkl', default='data/val.pickle')
    parser.add_argument('--val_anno_pkl', default='data/valannot.pickle')

    # lr, l2_reg, max_epoch, lr_decay, deacy_every,
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l2_reg', default=1e-4, type=float)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--num_answers', default=12, type=int)

    parser.add_argument('--model_from', default='model.pkl')
    parser.add_argument('--model_save', default='model.pkl')

    args = parser.parse_args()
    params = vars(args)

    main(params)
