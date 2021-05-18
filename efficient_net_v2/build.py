#!/usr/bin/env python

import time
from collections import Counter
from dataclasses import dataclass
from typing import Collection

import torch
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from detectron2.config import CfgNode, get_cfg
from detectron2.data import detection_utils
from detectron2.engine.defaults import AMPTrainer
from detectron2.data import detection_utils, DatasetCatalog, MetadataCatalog
from detectron2.data.build import build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)

from efficient_net_v2.model.backbone import build_effnet_backbone
from efficient_net_v2.datasets import PennFudanDataset, DatasetMapper


def register_dataset(
    dataset_root: str, dataset_name: str, is_train: bool = True
):
    DatasetCatalog.register(
        dataset_name, PennFudanDataset(root=dataset_root, is_train=is_train)
    )
    MetadataCatalog.get(dataset_name).set(
        thing_classes=['person'], thing_color=[(0, 255, 0)]
    )


def register_datasets(
    dataset_roots: Collection[str],
    dataset_names: Collection[str],
    is_train: bool = True,
):
    for dataset_root, dataset_name in zip(dataset_roots, dataset_names):
        register_dataset(dataset_root, dataset_name, is_train=is_train)


class AccumGradAMPTrainer(AMPTrainer):
    def __init__(self, accumulate: int = 1, *args, **kwargs):
        super(AccumGradAMPTrainer, self).__init__(*args, **kwargs)

        assert accumulate > 0, f'Accumulate must be >= 1 {accumulate}'
        self._accumulate = accumulate

    def run_step(self):
        cls_name = self.__class__.__name__

        assert (
            self.model.training,
            f'[{cls_name}] model was changed to eval mode!',
        )
        assert (
            torch.cuda.is_available(),
            f'[{cls_name}] CUDA is required for AMP Training',
        )

        start = time.perf_counter()
        data_time = time.perf_counter() - start

        loss_dicts = {}
        self.optimizer.zero_grad()

        for index in range(self._accumulate):
            data = next(self._data_loader_iter)
            loss_dict = self._run_one_step(data=data)
            if index == 0:
                loss_dicts = Counter(loss_dict)
                continue
            loss_dicts += Counter(loss_dict)

        self._write_metrics(loss_dicts, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def _run_one_step(self, data) -> dict:
        with autocast():
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {'total_loss': loss_dict}
            else:
                losses = sum(loss_dict.values())

        losses /= self._accumulate
        self.grad_scaler.scale(losses).backward()
        return loss_dict


@dataclass
class Trainer(DefaultTrainer):

    cfg: CfgNode = None
    is_train: bool = True

    def __post_init__(self):
        assert self.cfg
        super(Trainer, self).__init__(self.cfg)

        self._trainer = AccumGradAMPTrainer(
            accumulate=10,
            model=self.model,
            data_loader=self.data_loader,
            optimizer=self.optimizer,
        )

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()

    @classmethod
    def build_train_loader(cls, cfg: CfgNode, mapper=None):
        augmentations = cls.build_augmentation(cfg, True)
        if mapper is None:
            mapper = DatasetMapper(
                cfg=cfg,
                is_train=True,
            )
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_schedule(cls, cfg: CfgNode, optim: Optimizer):
        return build_d2_lr_schedule(cfg, optimizer)

    @classmethod
    def build_augmentation(cls, cfg: CfgNode, is_train: bool = True):
        result = detection_utils.build_augmentation(cfg, is_train=is_train)
        return result


def setup(args) -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    try:
        cfg.OUTPUT_DIR = args.output_dir
        cfg.MODEL.WEIGHTS = args.weights
    except AttributeError as e:
        print(e)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    register_datasets([args.root], cfg.DATASETS.TRAIN)
    register_datasets([args.root], cfg.DATASETS.TEST, is_train=False)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--root', required=False, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--num_gpus', required=False, type=int, default=1)
    parser.add_argument('--weights', required=False, type=str, default=None)
    args = parser.parse_args()
    print(args)

    train = True
    if train:
        launch(main, args.num_gpus, args=(args,))
    else:
        main(args)
