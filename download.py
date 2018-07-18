from torchvision import datasets, transforms

data = datasets.MNIST('data', train=True, download=True)
data = datasets.MNIST('data', train=False, download=True)
