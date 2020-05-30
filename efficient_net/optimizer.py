#!/usr/bin/env python

import os
import os.path as osp
import argparse
from datetime import datetime
import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('dark_background')

import tqdm
import torch
import torch.nn as nn
from torchvision import transforms

from efficient_net.network import EfficientNetX as EfficientNet
from efficient_net import appname, TrainConfig
from efficient_net.dataloader import CustomDataloader, create_loader
from efficient_net.utils import AverageMeter, ProgressMeter


def get_name(prefix=''):
    return prefix + '_' + datetime.now().strftime("%Y%m%d%H%M%S")


def create_logging_folder(config: TrainConfig):
    if not osp.isdir(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)
    log_dir = osp.join(config.LOG_DIR, get_name(appname()))
    os.mkdir(log_dir)
    return log_dir
        

class Optimizer(object):

    def __init__(self, config: TrainConfig):

        config.display()
        self.config = config
        model = EfficientNet(model_definition=config.MODEL)
        
        self._model = model
        
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
             transforms.Grayscale(num_output_channels=3),
             transforms.ColorJitter([0.5, 2], [0.5, 2], 0.5, 0.5),
             transforms.RandomAffine(degrees=180, scale=(0.2, 2.0)),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor()])
        
        train_ds, val_ds = create_loader(config.DATASET)
        self.train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        self.train_dl.transform = transform
        self.val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=config.VAL_BATCH_SIZE, shuffle=True)
        #self.val_dl.transform = transform

        self._criterion = nn.CrossEntropyLoss().to(self._device)
        self._optimizer = torch.optim.Adam(
            params=self._model.parameters(),
            lr=config.LR,
            # momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY)
        
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self._optimizer,
            step_size=config.LR_DECAY_EPOCH,
            gamma=config.LR_DECAY) if config.SCHEDULE_LR else None
        
        if config.WEIGHT_FILE is not None:
            checkpoint = torch.load(config.WEIGHT_FILE,
                                    map_location=self._device)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            # self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # self._model = nn.DataParallel(self._model)
        
        self._log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc@1',
            'train/acc@5',
            'valid/loss',
            'valid/acc@1',
            'valid/acc@5',
            'elapsed_time',
        ]
        
        self._log_dir = create_logging_folder(config=self.config)
        print(f'Data will be logged at {self._log_dir}')
                
        self._log_csv = osp.join(self._log_dir, 'log.csv')
        if not osp.exists(self._log_csv):
            with open(self._log_csv, 'w') as f:
                f.write(','.join(self._log_headers) + '\n')
                
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            
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

    def pbar_desc(
        self, name: str='Train', epoch: int=0, i: int=0,
        loss: float=-1, acc1: float=0.0, acc5: float=0.0):
        return f'{name.ljust(8)}' + \
          f'| epoch {epoch}/{self.config.EPOCHS} ' +\
          f'| iter {i:4} ' + \
          f'| Loss: {loss:.4f} ' +\
          f'| Acc@1 {acc1:.4f} ' +\
          f'| Acc@5 {acc5:.4f} |'
            
    def validate(self, epoch: int):
        self._model.to(self._device)
        self._model.eval()

        desc = f'Validatation'
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        progress = ProgressMeter(
            len(self.val_dl),
            [batch_time, data_time, losses, top1, top5],
            prefix="Val Epoch: [{}]".format(epoch))
                
        total_iter = len(self.val_dl)
        pbar = tqdm.tqdm(total=total_iter,
                         desc=self.pbar_desc(
                             name='Val', epoch=epoch))
        with torch.no_grad():
            start = time.time()
            total_correct = 0
            total_num = 0
            val_loss = 0
            
            for index, (images, labels) in enumerate(self.val_dl):
                images = images.to(self._device)
                labels = labels.to(self._device)
                prediction = self._model(images)
                
                # network loss
                loss = self._criterion(prediction, labels)
                loss_data = loss.data.item()
                val_loss += loss_data
            
                acc1, acc5 = Optimizer.accuracy(
                    prediction, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                desc = self.pbar_desc(
                    'Val', epoch, index, loss.item(), 
                    acc1[0], acc5[0])
                pbar.set_description(desc=desc, refresh=True)
                pbar.total = total_iter
                pbar.update(1)

                # update
                batch_time.update(time.time() - start)            
                start = time.time()

                if index == 4:
                    break

            pbar.close()
            # progress.display(0)
            
        result = dict(losses=losses, 
                      iteration=index,
                      top1=top1, top5=top5)
        return result
                  
    def train_one_epoch(self, epoch: int):
        
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
        
        start = time.time()        
        total_iter = len(self.train_dl)
        pbar = tqdm.tqdm(total=total_iter,
                         desc=self.pbar_desc(epoch=epoch))
        for index, (images, labels) in enumerate(self.train_dl):
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
            
            desc = self.pbar_desc(
                'Train', epoch, index, loss.item(), 
                acc1[0], acc5[0])
            pbar.set_description(desc=desc, refresh=True)
            pbar.total = total_iter
            pbar.update(1)
            
            loss_data = loss.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')

            loss.backward()
            self._optimizer.step()
            
            # update
            batch_time.update(time.time() - start)
            
            running_loss += loss_data
            start = time.time()

            if index == 5:
                break

        pbar.close()    
        # progress.display(index)

        result = dict(losses=losses, 
                      iteration=index,
                      top1=top1, top5=top5)        
        return result
    
    def optimize(self):

        if torch.cuda.is_available():
            self._model.cuda()

        self._model.to(self._device)
        self._model.train()

        plt.title("Learning Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss Score")
        fig, ax = plt.subplots()
        
        start_time = time.time()
        losses = []

        prev_tresult = None
        prev_vresult = None
        
        for epoch in range(self.config.EPOCHS):
            result = self.train_one_epoch(epoch)
            val_result = self.validate(epoch)
            
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
            
            with open(self._log_csv, 'a') as f:
                log = [epoch, 
                       result['iteration'], 
                       result['losses'].average, 
                       result['top1'].average,
                       result['top5'].average,
                       val_result['losses'].average, 
                       val_result['top1'].average,
                       val_result['top5'].average,
                       f'{time.time()-start_time:.6f}']
                log = map(str, log)
                f.write(','.join(log) + '\n')


            if prev_tresult is not None:
                xticks = np.array([epoch - 1, epoch])
                
                prev_loss = prev_tresult['losses'].average
                prev_std = prev_tresult['losses'].std
                prev_mean = prev_tresult['losses'].mean

                vprev_loss = prev_vresult['losses'].average
                vprev_std = prev_vresult['losses'].std
                vprev_mean = prev_vresult['losses'].mean

                losses = np.array([prev_loss, result['losses'].average])
                vlosses = np.array([vprev_loss, val_result['losses'].average])
                
                stds = np.array([prev_std, result['losses'].std])
                means = np.array([prev_mean, result['losses'].mean])
                vstds = np.array([vprev_std, val_result['losses'].std])
                vmeans = np.array([vprev_mean, val_result['losses'].mean])

                acc = np.array([prev_tresult['top1'].average,
                                result['top1'].average])
                vacc = np.array([prev_vresult['top1'].average,
                                val_result['top1'].average])
                
                plt.plot(xticks, losses, 'r-', label='Train Loss' if epoch == 1 else "")
                plt.plot(xticks, vlosses, 'b-', label='Val Loss' if epoch == 1 else "")
                # plt.plot(xticks, acc, 'g-', label='Train Acc' if epoch==1 else "")
                # plt.plot(xticks, vacc, 'y-', label='Val Acc' if epoch==1 else "")
                
                plt.fill_between(xticks, means - stds, means + stds, color='r',
                                 alpha=0.3, interpolate=True, lw=0.0)
                plt.fill_between(xticks, vmeans - vstds, vmeans + vstds, color='b',
                                 alpha=0.3, interpolate=True, lw=0.0)

                plt.legend(loc='best')
                plt.savefig(osp.join(self._log_dir, 'loss.png'), dpi=300)
            
            prev_tresult = result
            prev_vresult = val_result
            
            if epoch % self.config.SNAPSHOT_EPOCH == 0:
                model_name = os.path.join(
                    self._log_dir, self.config.SNAPSHOT_NAME + '.pt')
                torch.save(self._model.state_dict(), model_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'loss': result['losses']}, 
                    os.path.join(self._log_dir, 'checkpoint.pth.tar'))

            """
            del result
            if val_result is not None:
                del val_result
            """
            print('{s:{c}^{n}}'.format(s='', n=80, c='-'))
            
        
        
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
    parser.add_argument(
        '--lr', type=float, required=False, default=3e-4)
    parser.add_argument(
        '--decay', type=float, required=False, default=0.0)
    parser.add_argument(
        '--schedule_lr', action='store_true', default=False)
    
    args = parser.parse_args()

    config = TrainConfig()
    config.DATASET = args.dataset
    config.LOG_DIR = args.log_dir
    config.BATCH_SIZE = args.batch
    config.EPOCHS = args.epochs
    config.WEIGHT_FILE = args.weight
    config.LR = args.lr
    config.WEIGHT_DECAY = args.decay
    config.SCHEDULE_LR = args.schedule_lr
    
    o = Optimizer(config)
    o.optimize()
