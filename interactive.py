import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import datasets, transforms

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=0),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU())
            
        self.fc1 = nn.Linear(10 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, input):
    
        out = self.net(input)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out


def view_predictions(images, predictions, num_plot=8):

    if images.shape[0] < num_plot ** 2:
        return

    for i in range(num_plot):
    
        for j in range(num_plot):

            idx = i * num_plot + j

            plt.subplot(num_plot, num_plot, idx + 1)
            plt.imshow(images[idx])
            plt.axis('off')
    plt.show()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False)
    parser.add_argument('--epochs', dest='num_epochs', default=10, type=int)
    parser.add_argument('--lrate', dest='lrate', default=1e-2, type=float)
    parser.add_argument('--batch', dest='batch_size', default=128, type=int)

    kwargs = vars(parser.parse_args())

    use_cuda = kwargs['use_cuda'] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')


    train_data = datasets.MNIST('data', train=True, download=True,
                                transform=transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True,
                               transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_data, kwargs['batch_size'], True)
    test_loader = torch.utils.data.DataLoader(test_data, 1000, False)

    model = Network().to(device)

    optimizer = torch.optim.Adam(model.parameters(), kwargs['lrate'])

    for epoch in range(kwargs['num_epochs']):

        start = time.time()

        for batch in tqdm(train_loader, 'Training', leave=False):

            image, label = batch
            image = image.to(device)
            label = label.to(device)

            output = model(image)

            loss = nn.CrossEntropyLoss()(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total = 0.
        for batch in tqdm(test_loader, 'Testing ', leave=False):

            image, label = batch
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            _, pred = output.max(1, keepdim=True)
   
            correct = pred.eq(label.view_as(pred)).cpu().numpy().detach()

          
            total += correct.sum()

        print('Epoch %d accuracy: %2.4f, took: %2.4fs'%\
              (epoch, float(total) / float(len(test_data)), time.time() - start))


if __name__ == '__main__':
    main()
