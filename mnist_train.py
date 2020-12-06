"""Train a MNIST sparse model"""
import numpy as np
import argparse
from collections import OrderedDict
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def build_model(dims):
  l = len(dims)
  layers = []
  fcs = []
  masks = []

  for i in range(l - 2):
    w = nn.Linear(dims[i], dims[i + 1], bias=True)
    layers.append(('fc{}'.format(i + 1), w))
    fcs.append(w)
    masks.append(torch.ones((dims[i + 1], dims[i]), dtype=torch.float))
    layers.append(('relu{}'.format(i + 1), nn.ReLU()))
  w = nn.Linear(dims[l - 2], dims[l - 1], bias=True)
  layers.append(('fc{}'.format(l - 1), w))
  fcs.append(w)
  masks.append(torch.ones((dims[l - 1], dims[l - 2]), dtype=torch.float))
  model = nn.Sequential(OrderedDict(layers))

  return model, fcs, masks


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dims', '-n', type=str, required=True,
                      help='input_dim(784),layer1_out,...,layerk_out(10)')
  parser.add_argument('--sparsity', '-s', type=float,
                      help='Density of the weight matrix W and initial input vector')
  parser.add_argument('--iterations', '-i', type=int,
                      help='Number of training iterations')
  parser.add_argument('--epochs', '-e', type=int,
                      help='Number of training epochs')
  parser.add_argument('--batch_size', '-b', type=int, default=256,
                      help='Batch size')
  parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
  args = parser.parse_args()
  sparsity = args.sparsity
  iterations = args.iterations
  ratio = sparsity ** (1 / iterations)

  dims = args.dims.split(',')
  dims = [int(s) for s in dims]
  assert len(dims) >= 2

  trainset = MNIST(root='data', train=True, download=True,
                   transform=transforms.ToTensor())
  testset = MNIST(root='data', train=False, download=True,
                  transform=transforms.ToTensor())
  trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
  testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

  model, fcs, masks = build_model(dims)
  optimizer = optim.SGD(model.parameters(), lr=args.lr)
  criterion = nn.CrossEntropyLoss(reduction='mean')

  for i in range(iterations):
    print('Iteration {}'.format(i + 1))
    for e in range(args.epochs):
      train_epoch(model, trainloader, optimizer, criterion, fcs, masks)
    test(model, testloader)
    print('Pruning...')
    l = len(fcs)
    for j in range(l):
      w = fcs[j].weight.data.abs()
      m = masks[j]
      n = (m > 0).sum()
      d = int(n * (1 - ratio))
      p = w[m > 0].view(-1)
      r = p.sort().values[d]
      m[w < r] = 0
      masks[j] = m
      fcs[j].weight.data *= m
    test(model, testloader)

  p = [(fc.weight.data.numpy().transpose(), fc.bias.data.numpy()) for fc in fcs]
  with open('mnist.pkl', 'wb') as f:
    pickle.dump(p, f)

  # state = {'model': model.state_dict(),}
  # torch.save(state, 'mnist.pth')


def train_epoch(model, loader, optimizer, criterion, fcs, masks):
  model.train()
  for _, (inputs, targets) in enumerate(loader):
    inputs = inputs.view(-1, 784)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Prune
    l = len(fcs)
    for i in range(l):
      # print(fcs[i].weight.data.shape)
      # print(masks[i].shape)
      fcs[i].weight.data *= masks[i]


def test(model, loader):
  model.eval()
  n_correct = 0
  n_total = 0

  with torch.no_grad():
    for _, (inputs, targets) in enumerate(loader):
      inputs = inputs.view(-1, 784)
      n_total += len(inputs)
      outputs = model(inputs)
      predictions = torch.argmax(outputs, dim=1)
      n_correct += (predictions == targets).sum().item()

  print('Accuracy: {}'.format(n_correct / n_total))


if __name__ == '__main__':
  main()