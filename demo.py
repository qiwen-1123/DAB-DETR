import os, sys
import torch
import numpy as np

from models import build_DABDETR, build_dab_deformable_detr
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

# Init and Load Pre-trained Models
model_config_path = "model_zoo/DAB_DETR/R50/config.json" # change the path of the model config file
model_checkpoint_path = "model_zoo/DAB_DETR/R50/checkpoint.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path) 
model, criterion, postprocessors = build_DABDETR(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])

# Load Datasets
dataset_val = build_dataset(image_set='val', args=args)
cocojs = dataset_val.coco.dataset
id2name = {item['id']: item['name'] for item in cocojs['categories']}

image, targets = dataset_val[0]

# Visualize Model Predictions
output = model(image[None])
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]))[0]

thershold = 0.3 # set a thershold

scores = output['scores']
labels = output['labels']
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
select_mask = scores > thershold

box_label = [id2name[int(item)] for item in labels[select_mask]]
pred_dict = {
    'boxes': boxes[select_mask],
    'size': targets['size'],
    'box_label': box_label
}

vslzr = COCOVisualizer()
vslzr.visualize(image, pred_dict, savedir=None)