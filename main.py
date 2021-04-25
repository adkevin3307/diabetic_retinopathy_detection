import torch.nn as nn
import torch.optim as optim

from DataLoader import load_data
from Net import ResNet
from Model import Model

if __name__ == '__main__':
    train_set, test_set, train_loader, test_loader = load_data('/home/adkevin3307/Data/Datasets/Diabetic_Retinopathy/data', './csv_files')

    net = ResNet(pretrained=True)

    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    model = Model(net, optimizer, criterion)
    model.summary((1, 3, 224, 224))

    train_history = model.train(train_loader, epochs=10)
    test_history = model.test(test_loader)
