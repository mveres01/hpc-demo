import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import datasets, transforms


def view_predictions(images, predictions, num_plot=10, save_name=None):
    """Plots MNIST images with corresponding network predictions.
    
    Parameters
    ----------
    images: np.ndarray
        Numpy array of shape (num_images, channels, num_rows, num_cols)
    predictions: np.ndarray
        Numpy array of shape (num_predictions, 1)
    num_plot: int
        Number of predictions to plot on an image
    save_name: string
        Name of file we want to save image to
    """

    if images.shape[0] < num_plot:
        return

    fig = plt.figure()
    for i in range(num_plot):

        im = images[i, 0]

        plt.subplot(1, num_plot, i + 1)
        plt.imshow(im, cmap='gray')
        plt.axis('off')
        plt.title('%s' % predictions[i, 0])

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')


class Network(nn.Module):
    """Defines a 7-Layer CNN model."""

    def __init__(self):
        super(Network, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU())

        self.fc1 = nn.Linear(10 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, input):

        out = self.net(input) # conv, conv, pool, conv, conv, pool
        out = out.view(out.size(0), -1) 
        out = F.relu(self.fc1(out)) # fully-connected
        out = F.relu(self.fc2(out)) # fully-connected
        out = self.fc3(out) # fully-connected

        return out


def main():

    save_dir = 'images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False)
    parser.add_argument('--seed', dest='seed', default=1234, type=int)
    parser.add_argument('--epochs', dest='num_epochs', default=10, type=int)
    parser.add_argument('--lr', dest='lr', default=1e-2, type=float)
    parser.add_argument('--momentum', default=0.1, type=float)
    parser.add_argument('--batch', dest='batch_size', default=128, type=int)
    parser.add_argument('--no-progress', action='store_true', default=False)
    parser.add_argument('--optimizer', default='adam')
    kwargs = vars(parser.parse_args())

    np.random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])

    use_cuda = kwargs['use_cuda'] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # Define the dataset and how to sample items
    train_data = datasets.MNIST('data', train=True, download=True,
                                transform=transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True,
                               transform=transforms.ToTensor())

    # Define how to sample items from the dataset
    train_loader = torch.utils.data.DataLoader(train_data, 
                                               kwargs['batch_size'], True)
    test_loader = torch.utils.data.DataLoader(test_data, 1000, False)

    # Define the model
    model = Network().to(device)

    # Define how the network parameters will be updated
    if kwargs['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), kwargs['lr'])
    elif kwargs['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), kwargs['lr'], 
                                    momentum=kwargs['momentum'])
    else:
        raise NotImplementedError('Optimizer <%s>'%kwargs['optimizer'])

    # Start training the networks
    for epoch in range(kwargs['num_epochs']):

        start = time.time()

        for batch in tqdm(train_loader, 'Training', leave=False,
                          disable=kwargs['no_progress']):

            image, label = batch
            image = image.to(device)
            label = label.to(device)

            output = model(image)

            loss = nn.CrossEntropyLoss()(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = 0.
        first = False
        total = 0.

        for batch in tqdm(test_loader, 'Testing ', leave=False,
                          disable=kwargs['no_progress']):

            image, label = batch
            image = image.to(device)
            label = label.to(device)

            output = model(image)

            # Prediction is the element with maximum probability
            _, pred = output.max(1, keepdim=True)

            if first is False:
                first = True
                save_name = os.path.join(save_dir, 'epoch%d.jpg' % epoch)
                view_predictions(image.cpu().data.numpy(),
                                 pred.cpu().data.numpy(),
                                 num_plot=10, save_name=save_name)

            total += pred.eq(label.view_as(pred)).cpu().detach().sum()

        print('Epoch %d accuracy: %2.4f, took: %2.4fs' %
              (epoch, float(total) / float(len(test_data)), time.time() - start))


if __name__ == '__main__':
    main()
