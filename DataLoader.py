import os
import csv
import numpy as np
from PIL import Image
from typing import Callable, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_data(csv_root: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
    img_name, label = [], []

    with open(os.path.join(csv_root, mode + '_img.csv'), 'r') as csv_file:
        rows = csv.reader(csv_file)

        img_name = [row for row in rows]

    with open(os.path.join(csv_root, mode + '_label.csv'), 'r') as csv_file:
        rows = csv.reader(csv_file)

        label = [row for row in rows]

    return (np.squeeze(img_name), np.squeeze(label))


def load_data(root: str, csv_root: str) -> tuple[Dataset, Dataset, DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomRotation(90.0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_set = RetinopathyLoader(root, csv_root, 'train', train_transform)
    test_set = RetinopathyLoader(root, csv_root, 'test', test_transform)

    print(f'train: {len(train_set)}, test: {len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=16, num_workers=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, num_workers=8, shuffle=False)

    return (train_set, test_set, train_loader, test_loader)


class RetinopathyLoader(Dataset):
    def __init__(self, root: str, csv_root: str, mode: str, transform: Callable = None) -> None:
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = get_data(csv_root, mode)
        self.mode = mode
        self.transform = transform

        # print("> Found %d images..." % (len(self.img_name)))

    def __len__(self) -> int:
        """return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'

           step2. Get the ground truth label from self.label

           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping,
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints.

                  In the testing phase, if you have a normalization process during the training phase, you only need
                  to normalize the data.

                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]

            step4. Return processed image and label
        """

        filename = os.path.join(self.root, self.img_name[index] + '.jpeg')

        img = Image.open(filename)
        label = torch.tensor(int(self.label[index]), dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return (img, label)
