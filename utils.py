import argparse
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from RetinopathyDataset import RetinopathyDataset


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--root', type=str, required=True)
    parser.add_argument('-c', '--csv', type=str, required=True)
    parser.add_argument('-n', '--net', type=str, default='resnet18')
    parser.add_argument('-p', '--pretrained', action='store_true')
    parser.add_argument('-i', '--input_shape', type=int, nargs='+', default=[224, 224])
    parser.add_argument('-l', '--lr', type=float, default=1e-3)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-t', '--trainable', action='store_true')
    parser.add_argument('-L', '--load', type=str, default=None)
    parser.add_argument('-S', '--save', type=str, default=None)

    args = parser.parse_args()

    print('=' * 50)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('=' * 50)

    return args


def load_net(model_name: str, pretrained: bool = True) -> Union[models.ResNet, None]:
    net = None

    if model_name == 'resnet18':
        net = models.resnet18(pretrained=pretrained)
    if model_name == 'resnet50':
        net = models.resnet50(pretrained=pretrained)

    net.fc = nn.Linear(net.fc.in_features, 5)

    return net


def load_data(root: str, csv_root: str, shape: list) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomRotation((-30.0, 30.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(shape),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = RetinopathyDataset(root, csv_root, 'train', train_transform)
    test_set = RetinopathyDataset(root, csv_root, 'test', test_transform)

    print(f'train: {len(train_set)}, test: {len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=8, num_workers=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=8, shuffle=False)

    return (train_loader, test_loader)


def show_history(history: dict[str, list], name: str) -> None:
    plt.plot(history['loss'], label='train_loss')
    plt.plot(history['accuracy'], label='train_accuracy')

    if ('val_loss' in history) and ('val_accuracy' in history):
        plt.plot(history['val_loss'], label='valid_loss')
        plt.plot(history['val_accuracy'], label='valid_accuracy')

    plt.legend()
    plt.savefig(name)


def show_confusion_matrix(y_test: list, y_hat: list, name: str) -> None:
    labels = list(range(5))

    cm = confusion_matrix(y_test, y_hat, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot()
    plt.savefig(name)
