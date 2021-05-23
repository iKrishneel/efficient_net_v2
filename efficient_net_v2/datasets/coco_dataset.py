#!/usr/bin/env python

import os
import logging
from dataclasses import dataclass

from tqdm import tqdm

from detectron2.data import DatasetCatalog


logger = logging.getLogger(__name__)

LABELS = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
)


@dataclass
class CocoDataset(object):

    coco_names: str
    name: str = 'coco_2017_train'
    class_label = None

    def __post_init__(self):
        assert os.path.isfile(self.coco_names)

        with open(self.coco_names, 'r') as f:
            lines = f.read().split('\n')

        self.class_label = {}
        self.match_index = {}
        for i, line in enumerate(lines, 1):
            if line not in LABELS:
                continue
            self.class_label[line] = i
            self.match_index[i] = [len(self.match_index) + 1, line]

    def filter(self):
        logger.warning(f'Reading Catalog for {self.name}')
        coco_data = DatasetCatalog.get(self.name)

        self.dataset = []
        for data in tqdm(coco_data):
            annotations = []
            for anno in data['annotations']:
                category_id = anno['category_id']
                if category_id in self.class_label.values():
                    anno['category_id'] = self.match_index[category_id][0]
                    annotations.append(anno)
            if len(annotations) > 0:
                data['annotations'] = annotations
                self.dataset.append(data)

        logger.warning(f'New dataset size {len(self.dataset)}')
        assert len(self.dataset) > 0

    def __call__(self):
        self.filter()
        return self.dataset


if __name__ == '__main__':

    import sys
    c = CocoDataset(coco_names=sys.argv[1])

    import IPython
    IPython.embed()
