import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import parse, show_history
from DataLoader import load_data
from Net import ResNet
from Model import Model

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    args = parse()

    train_set, test_set, train_loader, test_loader = load_data(args.root, args.csv)

    net = ResNet(pretrained=args.pretrained)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    model = Model(net, optimizer, criterion)

    train_history = model.train(train_loader, epochs=args.epochs, val_loader=test_loader)
    test_history = model.test(test_loader)

    show_history(train_history)
