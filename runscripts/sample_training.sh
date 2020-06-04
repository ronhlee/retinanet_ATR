#!/usr/bin/env bash

/home/ronlee/anaconda3/envs/pytorch1.2/bin/python train.py \
--csv_train=inputs/train/synthetic.csv \
--csv_classes=inputs/sample_class.csv \
--depth=18 \
--epochs=5 \
--batch_size=32 \
--score_thresh=0.3 \
--iou_nms1=0.3 \
--lr=1e-5 \
--logfile=outputs/test_training/test.log
