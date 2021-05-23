#!/usr/bin/env python

import os
from dataclasses import dataclass
import numpy as np

import torch
from PIL import Image
import cv2 as cv


@dataclass
class PennFudanDataset(object):

    root: str = ""
    is_train: bool = True
    train_ratio: float = 0.8

    images = None
    masks = None

    def __post_init__(self):

        self.images = self.read_content('PNGImages')
        self.masks = self.read_content("PedMasks")
        assert len(self.images) == len(self.masks)

        np.random.seed(seed=256)
        indices = np.arange(0, len(self.images))
        np.random.shuffle(indices)

        train_size = int(len(self.images) * self.train_ratio)
        self.indices = (
            indices[:train_size] if self.is_train else indices[train_size:]
        )

        self.dataset = None

    def read_content(self, name: str) -> list:
        return list(sorted(os.listdir(os.path.join(self.root, name))))

    def __call__(self):
        self.dataset = []
        for index in self.indices:
            data = dict(
                file_name=os.path.join(
                    self.root, "PNGImages", self.images[index]
                ),
                sem_seg_file_name=os.path.join(
                    self.root, "PedMasks", self.masks[index]
                ),
                # image_id=int(self.images[index].split(os.sep)[-1].split('.')[0][8:]),
                image_id=index,
                height=800,
                width=800,
            )
            self.dataset.append(data)
        return self.dataset

    def get(self, index):
        # assert self.dataset and index < self.dataset
        dataset_dict = self.dataset[index]

        # image = Image.open(dataset_dict['file_name'])
        image = cv.imread(dataset_dict['file_name'], cv.IMREAD_ANYCOLOR)
        return image, dataset_dict

    def __getitem__(self, idx: int):
        im_path = os.path.join(self.root, "PNGImages", self.images[idx])
        mk_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        im = Image.open(im_path).convert("RGB")
        mk = Image.open(mk_path)
        mk = np.array(mk)

        # remove the background id
        obj_ids = np.unique(mk)[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mk == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin, ymin = np.min(pos[1]), np.min(pos[0])
            xmax, ymax = np.max(pos[1]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert to torch tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # dataset has only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = dict(
            boxes=boxes,
            labels=labels,
            masks=masks,
            image_id=image_id,
            area=area,
            iscrowd=iscrowd,
        )
        return im, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':

    import sys

    p = PennFudanDataset(root=sys.argv[1], is_train=False)
    print(p())
