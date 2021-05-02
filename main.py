import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import parse, load_net, load_data, show_history
from Model import Model

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    args = parse()

    train_loader, test_loader = load_data(args.root, args.csv, args.input_shape)

    net = load_net(args.net, args.pretrained)

    if args.load:
        net = torch.load(args.load)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    model = Model(net, optimizer, criterion)

    if args.trainable:
        history = model.train(train_loader, epochs=args.epochs, val_loader=test_loader)
        show_history(history)

    if args.save:
        model.save(args.save)

    model.test(test_loader)
