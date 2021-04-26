import torch.nn as nn
import torch.optim as optim

from utils import parse
from DataLoader import load_data
from Net import ResNet
from Model import Model

if __name__ == '__main__':
    args = parse()

    train_set, test_set, train_loader, test_loader = load_data(args.root, args.csv)

    net = ResNet(pretrained=args.pretrained)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    model = Model(net, optimizer, criterion)
    model.summary((1, 3, 224, 224))

    train_history = model.train(train_loader, epochs=args.epochs)
    test_history = model.test(test_loader)
