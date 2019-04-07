import torch
from torchvision import datasets, transforms
import tqdm

from model.capsnet import CapsNet
from model.loss import loss

def train(model, epochs=100, dataset='mnist', lr=0.001):

    torch.manual_seed(42)

    data_dir = "./data/"

    if dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.30801,))
                        ])),
            batch_size=64, shuffle=True)
    elif dataset == 'fashion-mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(data_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.30801,))
                        ])),
            batch_size=64, shuffle=True)
    else:
        print('Only accepts mnist | fashion-mnist')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            test_sample = data
            batch_size = test_sample.size()[0]
            # print(f"Sample size: {test_sample.size()}")
            output, reconstruction = model(data)
            L = loss(output, target, batch_size)
            L.backward()

            step = batch_idx + epoch
            if epoch % 10 == 0:
                tqdm.write(f'Epoch: {step}    Loss: {L.data.item()}')

            optimizer.step()

conv1_params = {
    "in_channels": 1,
    "out_channels": 256,
    "kernel_size": 9,
    "stride": 1
}

conv2_params = {
    "in_channels": 256,
    "out_channels": 8,
    "kernel_size": 9,
    "stride": 2
}

# NOTE. What parameters would we like to experiment with?
# num of capsules in PrimaryCaps? Capsule Dimensions? Conv params?
model = CapsNet(conv1_params, conv2_params)
train(model)