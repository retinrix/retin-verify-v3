#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = "yolox_idcard"
        self.output_dir = "./YOLOX_outputs"
        self.num_classes = 2
        self.data_dir = "./dataset_coco"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.mosaic_prob = 1.0
        self.mixup_prob = 0.5
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        self.warmup_epochs = 5
        self.max_epoch = 150
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 20
        self.min_lr_ratio = 0.05
        self.ema = True
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.save_history_ckpt = False
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.no_aug = False
        self.scale = (0.1, 2)
        self.shear = 2.0
        self.perspective = 0.0

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        import torch
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
