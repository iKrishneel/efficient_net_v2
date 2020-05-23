#!/usr/bin/env python

import os
import argparse
from datetime import datetime
import numpy as np
import time

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from efficient_net.network import EfficientNetX as EfficientNet
from efficient_net import appname, TrainConfig
from efficient_net.dataloader import CustomDataloader, create_loader
from efficient_net.utils import AverageMeter, ProgressMeter


def get_name(prefix=''):
    return prefix + '_' + datetime.now().strftime("%Y%m%d%H%M%S")


class Optimizer(object):

    def __init__(self, config: TrainConfig):

        config.display()
        self.config = config
        model = EfficientNet(model_definition=config.MODEL)
        # self._model = nn.DataParallel(model)
        self._model = model
        
        # logging directory setup
        if not os.path.isdir(config.LOG_DIR):
            os.mkdir(config.LOG_DIR)
        self._log_dir = os.path.join(config.LOG_DIR, get_name(appname()))
        os.mkdir(self._log_dir)

        # device
        self._device = torch.device(
            f'cuda:{config.DEVICE_ID}' \
            if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self._device}')
                
        total_params = 0
        for parameter in self._model.parameters():
            if parameter.requires_grad:
                total_params += np.prod(parameter.size())
        print('total_params:', total_params)

        # self.apply(self.weight_init)
        
        transform = transforms.Compose(
            [transforms.RandomResizedCrop(config.INPUT_SHAPE[1:]),
             # transforms.RandomCrop(config.INPUT_SHAPE[1:], pad_if_needed=True),
             transforms.ColorJitter(0.5, 0.5, 0, 0),
             transforms.RandomAffine(degrees=60, scale=(0.2, 2.0)),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor()])
        
        train_ds, val_ds = create_loader(config.DATASET)
        self.train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        self.train_dl.transform = transform
        self.val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=config.BATCH_SIZE//2, shuffle=True)

        self._criterion = nn.CrossEntropyLoss().to(self._device)
        self._optimizer = torch.optim.Adam(
            params=self._model.parameters(),
            lr=config.LR,
            # momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY * 0,)
        
        if config.WEIGHT_FILE is not None:
            checkpoint = torch.load(config.WEIGHT_FILE,
                                    map_location=self._device)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            # self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
    
        print(f'Current Device {torch.cuda.current_device()}')
        """
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self._optimizer,
            step_size=config.LR_DECAY_EPOCH,
            gamma=config.LR_DECAY)"""

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            
    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#putting-everything-together
    
    def validate(self):
        self._model.to(self._device)
        self._model.eval()

        desc = f'Validatation'
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        
        progress = ProgressMeter(
            len(self.train_dl),
            [batch_time, data_time, losses, top1, top5],
            prefix=desc)
        
        total_correct = 0
        total_num = 0
        val_loss = 0
        start = time.time()
        for images, labels in tqdm.tqdm(self.val_dl):
            images = images.to(self._device)
            labels = labels.to(self._device)
            prediction = self._model(images)
            
            # network loss
            loss = self._criterion(prediction, labels)
            loss_data = loss.data.item()
            val_loss += loss_data
            
            acc1, acc5 = Optimizer.accuracy(prediction, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

             # update
            batch_time.update(time.time() - start)            
            start = time.time()
            
        progress.display(0)
            
        val_loss /= len(self.val_dl)
        print(f'Validation loss: {val_loss}')
    
    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k
        """
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
        
    def train_one_epoch(self, epoch: int, prev_loss: float=0.0):
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        
        progress = ProgressMeter(
            len(self.train_dl),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        
        self._model.to(self._device)
        self._model.train()
        
        self.training = True
        running_loss = 0

        desc = f'epoch {epoch}/{self.config.EPOCHS}' + \
               f' Loss: {prev_loss}'
            
        start = time.time()
        batch_index = 0
        for images, labels in tqdm.tqdm(self.train_dl, desc=desc):
            data_time.update(time.time() - start)
            
            images = images.to(self._device)
            labels = labels.to(self._device)

            # clear gradients
            self._optimizer.zero_grad()

            prediction = self._model(images)
                
            # network loss
            loss = self._criterion(prediction, labels)
    
            acc1, acc5 = Optimizer.accuracy(prediction, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
    
            # loss /= len(images)
            loss_data = loss.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')

            loss.backward()
            self._optimizer.step()
            
            # update
            batch_time.update(time.time() - start)
            
            running_loss += loss_data
            batch_index += 1
            start = time.time()
            
        progress.display(batch_index)
        
        return running_loss  / batch_index # len(self.train_dl)
    
    def optimize(self):

        if torch.cuda.is_available():
            self._model.cuda()

        self._model.to(self._device)
        self._model.train()

        prev_loss = 0
        for epoch in tqdm.trange(self.config.EPOCHS,
                                 desc='efficient_net'):
            prev_loss = self.train_one_epoch(epoch, prev_loss)

            if epoch % 1 == 0:
                self.validate()
            
            # self._lr_scheduler.step()
            
            if epoch % self.config.SNAPSHOT_EPOCH == 0:
                model_name = os.path.join(
                    self._log_dir, self.config.SNAPSHOT_NAME + '.pt')
                torch.save(self._model.state_dict(), model_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'loss': prev_loss}, os.path.join(self._log_dir, 'checkpoint.pth.tar'))
                
        
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
    
    o = Optimizer(config)
    o.optimize()
    # o.validate()
