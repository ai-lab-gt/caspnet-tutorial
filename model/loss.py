import torch
from torch.autograd import Variable

def reconstruction_loss(v, target, image):
    pass

def margin_loss(v, target, batch_size):
    l = 0.5
    m = 0.9
    T = target.type(torch.FloatTensor)
    norm = torch.norm(v)
    zeros = Variable(torch.zeros(norm.size()))
    # L_k = T_k max(0, m^+ − ||v_k||)^2 + λ (1 − T_k) max(0, ||v_k|| − m^−)^2
    L = T * torch.max(zeros, m - norm) ** 2 + l * (1 -T) * torch.max(zeros, norm - (1. - m)) ** 2
    return torch.sum(L) / batch_size


def loss(v, target, batch_size):
    return margin_loss(v, target, batch_size)