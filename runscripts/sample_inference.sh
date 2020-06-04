#!/usr/bin/env bash

/home/ronlee/anaconda3/envs/pytorch1.2/bin/python visualize_fullscene.py \
inputs/sample_class.csv \
inputs/sample_inference_images.csv \
outputs/test_training/retinanet_2.pt \
outputs/test_inference \
--pix_overlap=300 \
--score_thresh=0.01 \
--iou_nms1=0.3 \
--iou_nms2=0.3 \
--detthresh=0.2 \
--fullscene=True

