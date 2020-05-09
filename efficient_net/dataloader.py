#!/usr/bin/env python

import os
import argparse
import numpy as np
from PIL import Image

import torch
from torchvision import transforms


def create_loader(dataset_dir: str, train_val_ratio: float = 0.7):

    np.random.seed(256)

    class_names = dict()
    train_ds = []
    val_ds = []
    for folder in os.listdir(dataset_dir):
        label, name = folder.split('.')
        path = os.path.join(dataset_dir, folder)
        label = int(label)

        class_names[label] = name

        image_files = os.listdir(path)

        s = int(train_val_ratio * len(image_files))
        x = image_files[:s]
        y = image_files[s:]

        for i in x:
            if len(i.split('.')) is not 2:
                continue
            im_path = os.path.join(path, i)
            train_ds.append([im_path, label])

        for i in y:
            if len(i.split('.')) is not 2:
                continue
            im_path = os.path.join(path, i)
            val_ds.append([im_path, label])

    return CustomDataloader(train_ds), CustomDataloader(val_ds)


class CustomDataloader(torch.utils.data.Dataset):

    def __init__(self, dataset: list,
                 input_shape=[3, 224, 224],
                 transform=None):
        super(CustomDataloader, )
        self.dataset = dataset
        self._input_shape = input_shape
        self._transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_path, label = self.dataset[idx]
        im = Image.open(im_path).convert('RGB')

        if self._transform is not None:
            im = self._transform(im)
        else:
            im = transforms.Resize(self._input_shape[1:])(im)
            im = transforms.ToTensor()(im)
        return [im, label]

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    train_dl, val_dl = create_loader(args.dataset)
    train_dl.transform = transforms.Compose(
        [transforms.CenterCrop((224, 224)),
         # transforms.Resize(input_shape[1:]),
         transforms.ColorJitter(0.5, 0.5, 0, 0),
         transforms.RandomAffine(degrees=30, scale=(0.5, 2.0)),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor()])

    l = torch.utils.data.DataLoader(train_dl, batch_size=1,
                                    shuffle=True)

    from IPython import embed
    embed()
