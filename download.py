import os
from torchvision import datasets, transforms

data_dir = 'data'
if not os.path.exists(data_dir):
     os.makedirs(data_dir)

data = datasets.MNIST(data_dir, train=True, download=True)
data = datasets.MNIST(data_dir, train=False, download=True)
