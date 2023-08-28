# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy
from util.misc import NestedTensor

from datasets.data_prefetcher import data_dict_to_cuda
from models.matcher import HungarianMatcher
from models.motr import TrainingTracker
from collections import deque
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler

from einops import rearrange

import gc



def preprocess_batch(data_dict: dict, matcher: HungarianMatcher):
    gt_ids = []
    bboxes = []
    scores = []
    for i in range(len(data_dict['imgs'])):
        gt_instances = data_dict['gt_instances'][i]
        proposals = data_dict['proposals'][i]
        gt_boxes = box_cxcywh_to_xyxy(gt_instances.boxes)
        gt_obj_ids = gt_instances.obj_ids
        proposals = proposals[proposals[:, 4] > 0.5]
        proposal_boxes = box_cxcywh_to_xyxy(proposals[..., :4])
        proposal_scores = proposals[..., 4]

        if gt_boxes.shape[0] == 0 or proposal_boxes.shape[0] == 0:
            gt_ids.append([])
            bboxes.append(torch.zeros((0, 4), device=gt_boxes.device))
            scores.append(torch.zeros((0, 1), device=gt_boxes.device))
            continue

        row, col = matcher(gt_boxes, proposal_boxes)

        gt_obj_ids = gt_obj_ids[row != -1]
        valid_idx = row[row != -1]
        gt_ids.append(gt_obj_ids.to(torch.long).tolist())

        matched_proposal_boxes = torch.clamp(proposal_boxes[valid_idx], min=0, max=1)
        matched_proposal_scores = proposal_scores[valid_idx].unsqueeze(1)
        
        bboxes.append(matched_proposal_boxes)
        scores.append(matched_proposal_scores)
     
    return bboxes, scores, gt_ids

# ONLY USED IF WE GET A BAD BATCH, extremely dirty
def clear_graph():
    for obj in gc.get_objects():
        try: # If it's a tensor (and not nn.Parameter), release it
            if isinstance(obj, torch.Tensor):
                obj.detach_()
                del obj
        except:
            pass
    torch.cuda.empty_cache()

def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    matcher: HungarianMatcher, device: torch.device, epoch: int, 
                    max_norm: float = 0, num_accumulate_batches: int = 1):
    model.train()
    criterion.train()

    training_tracker = TrainingTracker() 

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.0e}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=200, fmt='{value:.2f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=200, fmt='{value:.2f} ({avg:.4f})'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    grad_total_norm = 0
    acc_value = 0

    scaler = GradScaler()

    cnt = 0
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        try:
            training_tracker.reset()

            cnt += 1
            data_dict = data_dict_to_cuda(data_dict, device) 
            bboxes, scores, gt_ids = preprocess_batch(data_dict, matcher)
            
            imgs = torch.stack(data_dict['imgs'])

            samples = NestedTensor(
                imgs,
                torch.zeros((imgs.shape[0], imgs.shape[-2], imgs.shape[-1]), device=imgs.device)
            )
            
            lengths = torch.tensor([x.shape[0] for x in bboxes], device=imgs.device)
            padded_bboxes = torch.nn.utils.rnn.pad_sequence(bboxes, batch_first=True)
            attention_mask_bboxes = torch.nn.utils.rnn.pad_sequence([torch.zeros_like(x[:, 0]) for x in bboxes], batch_first=True, padding_value=1)
            padded_scores = torch.nn.utils.rnn.pad_sequence(scores, batch_first=True)

            if padded_scores.shape[1] == 0 or padded_bboxes.shape[1] == 0:
                continue
        
            with autocast(dtype=torch.bfloat16):
                queries = model(samples, padded_bboxes, padded_scores, attention_mask_bboxes)   

            unpadded_queries = torch.nn.utils.rnn.unpad_sequence(queries, lengths, batch_first=True)
            
            for q, ids in zip(unpadded_queries, gt_ids):
                training_tracker.update(q, ids)
            
            tracks, observation_mask = training_tracker.get_tracks() 

            with autocast(dtype=torch.bfloat16):
                predictions = model.predictor(tracks)
            
            cum_mask, _ = torch.cummax(observation_mask, dim=1)
            mask = cum_mask[:, :-1] & observation_mask[:, 1:]

            predictions = predictions[:, :-1]
            targets = tracks[:, 1:]

            predictions = rearrange(predictions, 'b n d -> n b d')
            targets = rearrange(targets, 'b n d -> n b d')
            mask = rearrange(mask, 'b n -> n b')
            
            losses = []
            accs = []
            for p, t, m, in zip(predictions, targets, mask): 
                p = p[m]
                t = t[m]
                if p.shape[0] == 0 or t.shape[0] == 0:
                    assert p.shape[0] == t.shape[0]
                    continue
                with autocast(dtype=torch.bfloat16):
                    l = criterion(p, t)
                losses.append(l)
                a = criterion.cosine_sim_accuracy(p, t)
                accs.append(a)
            
            loss = torch.stack(losses).mean()
            acc = torch.stack(accs).mean()

            loss_value = loss.item()
            
            loss = loss / num_accumulate_batches
            acc_value = acc.item()

            scaler.scale(loss).backward()
            
            if cnt % num_accumulate_batches == 0:
                scaler.unscale_(optimizer)
                if max_norm > 0:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                else:
                    grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(grad_norm=grad_total_norm)
            metric_logger.update(acc=acc_value)
        except Exception as e:
            # Don't ignore keyboard interrupts.
            if isinstance(e, KeyboardInterrupt):
                raise e
            torch.cuda.empty_cache()
            continue

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
