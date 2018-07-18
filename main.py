import time
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def main():

    num_epochs = 10
    batch_size = 128
    lrate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu') 

    train_data = datasets.MNIST('data', train=True, download=True,
                                transform=transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True,
                               transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, True)
    test_loader = torch.utils.data.DataLoader(test_data, 1000, False)

    model = Network().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lrate)

    for epoch in range(num_epochs):

        start = time.time()

        for batch in train_loader:

            image, label = batch
            image = image.to(device)
            label = label.to(device)

            output = model(image)

            loss = nn.CrossEntropyLoss()(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = 0.
        for batch in test_loader:

            image, label = batch
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            _, pred = output.max(1, keepdim=True)
   
            correct += pred.eq(label.view_as(pred)).sum().detach()
            

        print('Epoch %d accuracy: %2.4f, took: %2.4fs'%\
              (epoch, float(correct) / float(len(test_data)), time.time() - start))



if __name__ == '__main__':
    main()
