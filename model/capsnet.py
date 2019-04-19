import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
# from tensorboardX import SummaryWriter
from tqdm import *

def squash(input):
    """
    Squashing function for a tensor.
    :param input: torch.Tensor
    """
    assert (input.norm() > 0), "Division by zero in second term of equation"
    norm = input.norm()
    squared_norm = norm.pow(2)
    return (squared_norm/(1+squared_norm))*(input/norm)


class PrimaryCapsules(nn.Module):
    """
    Primary Capsule Network on MNIST.
    :param conv1_params: Parameters for first Conv2d layer
    :param conv2_params: Parameters for second Conv2d layer
    :param caps_maps: number of feature maps (capsules)
    :param caps_dims: dimension of each capsule's activation vector
    Shape:
        - Input: (batch, channels, height, width)
        - Output: (batch, n_caps, caps_dims)
    """
    def __init__(self, conv_params, caps_maps=32, num_capsules=8):
        super(PrimaryCapsules, self).__init__()
        # Output of conv2 has 256 (32*8) maps of 6x6.
        # We instead want 32 vectors of 8 dims each.
        self.num_routes = caps_maps * 6 * 6
        self.num_capsules = num_capsules

        self.capsules = nn.ModuleList([
            nn.Conv2d(**conv_params) for _ in range(self.num_capsules)
        ])

    def forward(self, x):
        output = [capsule(x) for capsule in self.capsules]
        output = torch.stack(output, dim=1)
        output = output.view(x.size(0), self.num_routes, -1)
        return squash(output)

# https://github.com/laubonghaudoi/CapsNet_guide_PyTorch/blob/master/DigitCaps.py
class DigitCapsule(nn.Module):
    """
    Digit Capsule Layer.
    :param num_lower_capsules: Number of lower level capsules, used to calculate dynamic routing.
    Shape:
        - Input: (batch, channels, height, width)
        - Output: (batch, n_caps, caps_dims)
    """
    def __init__(self, num_route_nodes, in_channels, out_channels, num_iterations):
        super(DigitCapsule, self).__init__()
        self.num_capsules = 10
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        # W.shape => [1, 10, 1152, 8, 16]
        # (1) is to be broadcastable with torch.matmul
        self.W = nn.Parameter(torch.randn(1,
                                        self.num_route_nodes,
                                        self.num_capsules,
                                        out_channels,
                                        in_channels,
                                        ))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        # Routing Algorithm.
        # for all capsule i in layer l and capsule j in layer (l + 1): b_ij ← 0
        b = Variable(torch.zeros(1, self.num_route_nodes, self.num_capsules, 1))
        # for r iterations do
        for i in range(self.num_iterations):
            # for all capsule i in layer l: c_i ← softmax(b_i)
            c = F.softmax(b)
            c = torch.cat([c] * batch_size, dim=0).unsqueeze(4)
            # for all capsule j in layer (l+1): s_j ← SUM_i ( c_ij * u_hat_j|i )
            s = (u_hat * c).sum(dim=1, keepdim=True)
            # for all capsule j in layer (l + 1): v_j ← squash(s_j)
            v = squash(s)
            # for all capsule i in layer l and capsule j in layer (l + 1): b_ij ← b_ij + u_hat_j|i * v_j
            if i < self.num_iterations - 1:
                a = torch.matmul(u_hat.transpose(3, 4), torch.cat([v] * self.num_route_nodes, dim=1))
                b = b + a.squeeze(4).mean(dim=0, keepdim=True)

        return v.squeeze(1)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )
        
    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes)
        
        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        
        reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 1, 28, 28)
        
        return reconstructions, masked

class CapsNet(nn.Module):
    def __init__(self, conv1_params, conv2_params):
        super(CapsNet, self).__init__()
        self.conv = nn.Conv2d(**conv1_params)
        self.primary_capsules = PrimaryCapsules(conv2_params, num_capsules=8)
        self.digit_capsules = DigitCapsule(num_route_nodes=32*6*6, in_channels=8, out_channels=16, num_iterations=3)
        self.decoder = Decoder()
        # self.decoder = nn.Sequential(
        #     nn.Linear(16 * 10, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 784),
        #     nn.Sigmoid()
        # ) 

    def forward(self, x):
        x = F.relu(self.conv(x))
        u = self.primary_capsules(x)
        v = self.digit_capsules(u)
        print('V size', v.size())
        reconstruction, masked = self.decoder(v, x)
        print('reconstruction size', reconstruction.size())
        return v, reconstruction