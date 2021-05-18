#!/usr/bin/env python

import os
import os.path as osp
import argparse
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms

from yacs.config import CfgNode as CN

from efficient_net_v2.config import get_cfg
from efficient_net_v2.model import EfficientNetV2
from efficient_net_v2.logger import logger


def get_transforms() -> dict:
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return {
        'train': transforms.Compose(
            [
                transforms.Resize([224, 224]),
                # transforms.RandomResizedCrop(config.INPUT_SHAPE[1:]),
                # transforms.Grayscale(num_output_channels=3),
                # transforms.ColorJitter([0.5, 2], [0.5, 2], 0.5, 0.5),
                # transforms.RandomAffine(degrees=180, scale=(0.2, 2.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                norm,
            ]
        ),
        'val': transforms.Compose(
            [
                # transforms.Resize(config.INPUT_SHAPE[1:]),
                transforms.ToTensor(),
                norm,
            ]
        ),
    }


class Coco2017Dataset(torch.utils.data.Dataset):
    def __init__(
        self, root: str, data_type='train', num_class: int = 91, **kwargs: dict
    ):
        self.data_transforms = get_transforms()
        file_name = 'train2017' if data_type == 'train' else 'val2017'
        self.dataset = CocoDetection(
            root=osp.join(root, file_name),
            annFile=osp.join(root, f'annotations/instances_{file_name}.json'),
            # transform=data_transforms[data_type],
        )

        """
        logger.info(f'Dataset DRY Run: {len(dataset)}')
        self.dataset = []
        for d in tqdm.tqdm(dataset):
            if len(d[1]) == 0:
                continue
            self.dataset.append(d)
        """

        self.data_type = data_type
        self.num_class = num_class
        np.random.seed(256)

    def __getitem__(self, index: int, use_cropped: bool = True):
        if use_cropped:
            return self.get_cropped_instance(index)

        image, targets = self.dataset.__getitem__(index)

        labels = torch.zeros(self.num_class, dtype=torch.int64)
        for target in targets:
            cat_id = target['category_id']
            labels[cat_id] = cat_id
        return image, labels

    def get_cropped_instance(self, index: int):

        while True:
            image, targets = self.dataset.__getitem__(index)
            if len(targets) > 0:
                break
            index = np.random.randint(0, len(self.dataset))

        selected = np.random.randint(0, len(targets))
        target = targets[selected]

        category_id = target['category_id']
        bbox = np.array(target['bbox'], dtype=np.intp)
        center = np.average([bbox[:2], bbox[:2] + bbox[2:]], axis=0)

        new_size = np.array(
            (
                np.random.randint(bbox[2], 2 * np.maximum(bbox[2], 1)),
                np.random.randint(bbox[3], 2 * np.maximum(bbox[3], 1)),
            ),
        )

        x1, y1 = np.maximum(center - new_size / 2, [0, 0]).astype(np.intp)
        x2, y2 = np.minimum(center + new_size / 2, image.size).astype(np.intp)
        image = image.crop([x1 + 1, y1 + 1, x2 - 1, y2 - 1])

        return self.data_transforms[self.data_type](image), category_id

    def __len__(self):
        return len(self.dataset)


def get_name(prefix=''):
    return prefix + '_' + datetime.now().strftime("%Y%m%d%H%M%S")


def create_logging_dir(output_dir):
    if not osp.isdir(output_dir):
        os.mkdir(output_dir)
    log_dir = osp.join(output_dir, get_name('train'))
    os.mkdir(log_dir)
    return log_dir


@dataclass
class Optimizer(object):

    cfg: CN
    train_dl: DataLoader
    val_dl: DataLoader = None

    def __post_init__(self):
        assert self.cfg is not None
        assert self.train_dl is not None

        self.device = torch.device(
            f'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f'Device: {self.device}')

        self.model = EfficientNetV2(self.cfg)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # self.criterion = nn.BCELoss()
        self.optim = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.cfg.SOLVER.LR,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optim,
            step_size=self.cfg.SOLVER.LR_DECAY_STEP,
            gamma=self.cfg.SOLVER.LR_DECAY,
        )

        total_params = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                total_params += np.prod(parameter.size())
        logger.info(f'total_params: {total_params}')

        self.log_dir = None

    def run(self):
        self()

    def __call__(self):
        epochs = self.cfg.SOLVER.EPOCHS
        lr = self.cfg.SOLVER.LR
        batch = self.cfg.SOLVER.BATCH_SIZE

        epoch_loss = ''
        with tqdm.tqdm(total=epochs, desc='Epochs %s') as epoch_pbar:
            for epoch in range(epochs):
                epoch_pbar.set_description(
                    desc=f'Epoch {epoch + 1} | LOSS: {epoch_loss} '
                    + f'| LR: {lr} | BATCH: {batch}',
                    refresh=True,
                )
                epoch_pbar.update(1)
                epoch_loss = self.train_one_epoch(epoch=epoch)

                if self.log_dir is None:
                    self.log_dir = create_logging_dir(self.cfg.OUTPUT_DIR)
                    logger.info(f'Output Directory: {self.log_dir}')

                torch.save(
                    {
                        'model_state_dict': self.model.state_dict(),
                    },
                    osp.join(self.log_dir, f'checkpoint.pth.tar'),
                )

    def train_one_epoch(self, epoch):
        self.model.train()

        running_loss = 0.0
        total_iter = len(self.train_dl)
        with tqdm.tqdm(total=total_iter) as pbar:
            for index, (images, targets) in enumerate(self.train_dl):

                images = images.to(self.device)
                targets = targets.to(self.device)

                self.optim.zero_grad()
                result = self.model(images)
                loss = self.criterion(result, targets)

                # loss_data = loss.data()
                running_loss += loss
                assert not (torch.isnan(loss) or torch.isinf(loss))

                loss.backward()
                self.optim.step()

                pbar.set_description(desc='Loss %3.8f' % loss)
                pbar.update(1)

        return running_loss / len(self.train_dl)

    def validation(self):

        self.model.eval()
        with torch.no_grad():
            pass

    def load_state_dict(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=strict)


def main(args):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = args.output_dir

    num_workers = cfg.DATASETS.NUM_WORKER
    batch_size = cfg.SOLVER.BATCH_SIZE

    train_dl = DataLoader(
        Coco2017Dataset(root=args.root),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_dl = DataLoader(
        Coco2017Dataset(root=args.root, data_type='val'),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    optimizer = Optimizer(
        cfg=cfg,
        train_dl=train_dl,
    )  # val_dl=val_dl)
    optimizer()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    main(args)
