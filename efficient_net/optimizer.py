#!/usr/bin/env python

import os
import argparse
from datetime import datetime
import numpy as np

import tqdm
import torch
import torch.nn as nn
from torchvision import transforms

from efficient_net.network import EfficientNetX as EfficientNet
from efficient_net import appname, TrainConfig
from efficient_net.dataloader import CustomDataloader, create_loader


def get_name(prefix=''):
    return prefix + '_' + datetime.now().strftime("%Y%m%d%H%M%S")


class Optimizer(EfficientNet):

    def __init__(self, config: TrainConfig):
        super(Optimizer, self).__init__(model_definition=config.MODEL)
        config.display()
        self.config = config

        # logging directory setup
        if not os.path.isdir(config.LOG_DIR):
            os.mkdir(config.LOG_DIR)
        self._log_dir = os.path.join(config.LOG_DIR, get_name(appname()))
        os.mkdir(self._log_dir)

        # device
        self._device = torch.device(
            f'cuda:{config.DEVICE_ID}'
            if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self._device}')

        self._criterion = nn.CrossEntropyLoss().to(self._device)
        self._optimizer = torch.optim.RMSprop(
            params=self.parameters(),
            lr=config.LR,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,)

        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self._optimizer,
            step_size=config.LR_DECAY_EPOCH,
            gamma=config.LR_DECAY)

        total_params = 0
        for parameter in self.parameters():
            if parameter.requires_grad:
                total_params += np.prod(parameter.size())
        print('total_params:', total_params)

        transform = transforms.Compose(
            [transforms.CenterCrop(config.INPUT_SHAPE[1:]),
             transforms.ColorJitter(0.5, 0.5, 0, 0),
             transforms.RandomAffine(degrees=30, scale=(0.5, 2.0)),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor()])

        train_ds, val_ds = create_loader(config.DATASET)
        self.train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        self.val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=config.BATCH_SIZE, shuffle=True)

        if config.WEIGHT_FILE is not None:
            self.load_state_dict(torch.load(config.WEIGHT_FILE))

    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#putting-everything-together

    def train_one_epoch(self, epoch: int):
        self.train()
        running_loss = 0

        desc = f'epoch {epoch}/{self.config.EPOCHS}' + \
               f'Loss: {prev_loss}'
        for i in tqdm.trange(self.config.ITER_PER_EPOCH, desc=desc):

            # dataloader
            images, labels = next(iter(self.train_dl))
            images = images.to(self._device)
            labels = labels.to(self._device)

            # clear gradients
            self._optimizer.zero_grad()

            # propagate the data
            prediction = self.__call__(images)

            # network loss
            loss = self._criterion(prediction, labels)
            loss /= len(images)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')

            # update network weight
            loss.backward()
            self._optimizer.step()

            running_loss += loss_data

    def optimize(self):

        if torch.cuda.is_available():
            self.cuda()

        self.to(self._device)
        self.train()

        prev_loss = 0
        for epoch in tqdm.trange(self.config.EPOCHS,
                                 desc='efficient_net'):
            running_loss = 0.0
            desc = f'epoch {epoch}/{self.config.EPOCHS}' + \
                   f'Loss: {prev_loss}'
            for i in tqdm.trange(self.config.ITER_PER_EPOCH,
                                 desc=desc):

                # dataloader
                images, labels = next(iter(self.train_dl))
                images = images.to(self._device)
                labels = labels.to(self._device)

                # clear gradients
                self._optimizer.zero_grad()

                # propagate the data
                prediction = self.__call__(images)

                # network loss
                loss = self._criterion(prediction, labels)
                loss /= len(images)
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while training')

                # update network weight
                loss.backward()
                self._optimizer.step()

                running_loss += loss_data

            self._lr_scheduler.step()
            prev_loss = running_loss / self.config.ITER_PER_EPOCH

            if epoch % self.config.SNAPSHOT_EPOCH == 0:
                model_name = os.path.join(
                    self._log_dir, self.config.SNAPSHOT_NAME + '.pt')
                torch.save(self.state_dict(), model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true', default=True)
    parser.add_argument(
        '--dataset', type=str, required=True)
    parser.add_argument(
        '--log_dir', type=str, required=False, default='logs')
    parser.add_argument(
        '--batch', type=int, required=False, default=1)
    parser.add_argument(
        '--epochs', type=int, required=False, default=100)
    parser.add_argument(
        '--weight', type=str, required=False, default=None)

    args = parser.parse_args()

    config = TrainConfig()
    config.DATASET = args.dataset
    config.LOG_DIR = args.log_dir
    config.BATCH_SIZE = args.batch
    config.EPOCHS = args.epochs
    config.WEIGHT_FILE = args.weight

    o = Optimizer(config).optimize()
