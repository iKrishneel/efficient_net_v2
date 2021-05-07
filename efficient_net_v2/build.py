#!/usr/bin/env python

from detectron2.config import CfgNode, get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch
)

from efficient_net_v2.model.backbone import build_effnet_backbone


def setup(args) -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    try:
        cfg.OUTPUT_DIR = args.output_dir
        cfg.MODEL.WEIGHTS = args.weights
    except AttributeError as e:
        pass

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == '__main__':
    parser = default_argument_parser()
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

