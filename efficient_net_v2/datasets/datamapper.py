#!/usr/bin/env python

from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch

from detectron2.config import CfgNode as CN
from detectron2.data import transforms as T
from detectron2.data import detection_utils as dutils

from pycocotools.mask import encode


@dataclass
class DatasetMapper(object):

    cfg: CN = None
    is_train: bool = True

    def __post_init__(self):
        assert self.cfg
        self._augmentation = self.build_augmentation()

    def __call__(self, dataset_dict: dict):
        dataset_dict = deepcopy(dataset_dict)

        image = dutils.read_image(
            dataset_dict.get('file_name'), format=self.cfg.INPUT.FORMAT
        )
        mask = dutils.read_image(
            dataset_dict.pop('sem_seg_file_name'),
        )

        assert image.shape[:2] == mask.shape[:2]

        obj_ids = np.unique(mask)[1:]
        masks = mask == obj_ids[:, None, None]

        annotations = []
        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            box = (
                np.min(pos[1]),
                np.min(pos[0]),
                np.max(pos[1]),
                np.max(pos[0]),
            )
            annotations.append(
                {
                    'bbox': box,
                    'bbox_mode': 0,
                    'category_id': 0,
                    'segmentation': encode(
                        np.array(mask, dtype=np.uint8, order='F')
                    ),
                }
            )

        if not self.is_train:
            return dict(
                image=image,
                annotations=annotations
            )

        aug_input = T.AugInput(image, sem_seg=mask)
        transforms = aug_input.apply_augmentations(self._augmentation)
        image = torch.from_numpy(
            aug_input.image.transpose((2, 0, 1)).astype('float32')
        )
        mask = torch.from_numpy(aug_input.sem_seg.astype('float32'))

        annos = [
            dutils.transform_instance_annotations(
                annotation, transforms, image.shape[1:]
            )
            for annotation in annotations
        ]

        instances = dutils.annotations_to_instances(
            annos, image.shape[1:], mask_format=self.cfg.INPUT.MASK_FORMAT
        )
        # instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        dataset_dict['image'] = image
        dataset_dict['sem_seg'] = mask
        # dataset_dict['instances'] = instances[instances.gt_boxes.nonempty()]
        dataset_dict['instances'] = dutils.filter_empty_instances(instances)
        return dataset_dict

    def build_augmentation(self):
        result = dutils.build_augmentation(self.cfg, is_train=self.is_train)
        if self.is_train:
            # resize = T.Resize((800, 800))
            # result.extend([resize, ])
            pass
        return result
