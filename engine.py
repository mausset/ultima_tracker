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
from models.motr import ContrastiveCriterion, TrainingTracker
from collections import deque
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler

from einops import rearrange


def preprocess_batch(data_dict: dict, matcher: HungarianMatcher):
    gt_ids = []
    bboxes = []
    matched_masks = []
    scores = []

    has_matched = False
    for i in range(len(data_dict['imgs'])):
        gt_instances = data_dict['gt_instances'][i]
        proposals = data_dict['proposals'][i]
        gt_boxes = box_cxcywh_to_xyxy(gt_instances.boxes)
        gt_obj_ids = gt_instances.obj_ids
        proposals = proposals[proposals[:, 4] > 0.05]
        proposal_boxes = box_cxcywh_to_xyxy(proposals[..., :4])
        proposal_scores = proposals[..., 4]

        if gt_boxes.shape[0] == 0 or proposal_boxes.shape[0] == 0:
            gt_ids.append([])
            bboxes.append(torch.zeros((0, 4), device=gt_boxes.device))
            scores.append(torch.zeros((0, 1), device=gt_boxes.device))
            matched_masks.append(torch.zeros((0, 1), device=gt_boxes.device))
            continue

        row, col = matcher(gt_boxes, proposal_boxes)

        gt_obj_ids = gt_obj_ids[row != -1]
        gt_ids.append(gt_obj_ids.to(torch.long).tolist())
        valid_idx = row[row != -1]
        if valid_idx.shape[0] != 0:
            has_matched = True

        matched_proposal_boxes = torch.clamp(proposal_boxes[valid_idx], min=0, max=1)
        matched_proposal_scores = proposal_scores[valid_idx].unsqueeze(1)

        unmatched_proposal_boxes = torch.clamp(proposal_boxes[col == -1], min=0, max=1)
        unmatched_proposal_scores = proposal_scores[col == -1].unsqueeze(1)

        box = torch.cat([matched_proposal_boxes, unmatched_proposal_boxes], dim=0)
        score = torch.cat([matched_proposal_scores, unmatched_proposal_scores], dim=0)
        mask = torch.cat([torch.ones_like(matched_proposal_scores), torch.zeros_like(unmatched_proposal_scores)], dim=0).to('cuda')
        
        bboxes.append(box)
        scores.append(score)
        matched_masks.append(mask)
     
    return bboxes, scores, gt_ids, matched_masks, has_matched

def train_one_epoch_mot(model: torch.nn.Module, criterion: ContrastiveCriterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    matcher: HungarianMatcher, device: torch.device, epoch: int, 
                    max_norm: float = 0, num_accumulate_batches: int = 1):
    model.train()
    criterion.train()

    training_tracker = TrainingTracker() 

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.0e}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=500, fmt='{value:.2f}'))
    metric_logger.add_meter('assoc_acc', utils.SmoothedValue(window_size=500, fmt='{value:.2f} ({avg:.4f})'))
    metric_logger.add_meter('conf_acc', utils.SmoothedValue(window_size=500, fmt='{value:.2f} ({avg:.4f})'))
    metric_logger.add_meter('pos_acc', utils.SmoothedValue(window_size=500, fmt='{value:.2f} ({avg:.4f})'))
    metric_logger.add_meter('neg_acc', utils.SmoothedValue(window_size=500, fmt='{value:.2f} ({avg:.4f})'))
    metric_logger.add_meter('ctr_loss', utils.SmoothedValue(window_size=500, fmt='{value:.4f} ({avg:.4f})'))
    metric_logger.add_meter('conf_loss', utils.SmoothedValue(window_size=500, fmt='{value:.4f} ({avg:.4f})'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    grad_total_norm = 0
    acc_value = 0

    scaler = GradScaler()

    cnt = 0
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        training_tracker.reset()

        cnt += 1
        data_dict = data_dict_to_cuda(data_dict, device)
        bboxes, scores, gt_ids, matched_masks, has_matched = preprocess_batch(data_dict, matcher)

        if not has_matched:
            continue
        
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
            queries, refined_scores = model(samples, padded_bboxes, padded_scores, attention_mask_bboxes)

            unpadded_refined_scores = torch.nn.utils.rnn.unpad_sequence(refined_scores, lengths, batch_first=True)

            conf_losses = []
            pos_accs = []
            neg_accs = []
            conf_accs = []
            for i in range(len(unpadded_refined_scores)):
                if unpadded_refined_scores[i].shape[0] == 0:
                    continue
                
                gts = matched_masks[i]

                rs = unpadded_refined_scores[i]
                conf_l = criterion.confidence_loss(rs, gts)
                conf_losses.append(conf_l)
                acc, pos_acc, neg_acc = criterion.confidence_accuracy(rs, gts)
                pos_accs.append(pos_acc)
                neg_accs.append(neg_acc)
                conf_accs.append(acc)
        
        conf_loss = torch.stack(conf_losses).nanmean() if len(conf_losses) > 0 else torch.zeros((1,), device=loss.device)
        pos_acc = torch.stack(pos_accs).nanmean()
        neg_acc = torch.stack(neg_accs).nanmean()
        conf_acc = torch.stack(conf_accs).nanmean()
    
        unpadded_queries = torch.nn.utils.rnn.unpad_sequence(queries, lengths, batch_first=True)
        unpadded_queries = [x[mask.bool().squeeze(1)] for x, mask in zip(unpadded_queries, matched_masks)]

        for q, ids in zip(unpadded_queries, gt_ids):
            training_tracker.update(q, ids)
        
        tracks, observation_mask = training_tracker.get_tracks() 

        cum_mask, _ = torch.cummax(observation_mask, dim=1)
        mask = cum_mask[:, :-1] & observation_mask[:, 1:]

        key_padding_mask = ~cum_mask.to(device)
        with autocast(dtype=torch.bfloat16):
            predictions = model.predictor(tracks, src_key_padding_mask=key_padding_mask)

        
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
        
        contrastive_loss = torch.stack(losses).mean() if len(losses) > 0 else torch.zeros((1,), device=loss.device)
        
        loss = conf_loss + contrastive_loss
        assoc_acc = torch.stack(accs).mean() if len(accs) > 0 else torch.zeros((1,), device=loss.device)

        loss_value = loss.item() if not math.isnan(loss.item()) else 0
        ctr_loss_value = contrastive_loss.item() if not math.isnan(contrastive_loss.item()) else 0
        conf_loss_value = conf_loss.item() if not math.isnan(conf_loss.item()) else 0

        loss = loss / num_accumulate_batches
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
        metric_logger.update(assoc_acc=assoc_acc.item())
        metric_logger.update(conf_acc=conf_acc.item())
        if not math.isnan(pos_acc.item()):
            metric_logger.update(pos_acc=pos_acc.item())
        if not math.isnan(neg_acc.item()):
            metric_logger.update(neg_acc=neg_acc.item())
        metric_logger.update(ctr_loss=ctr_loss_value)
        metric_logger.update(conf_loss=conf_loss_value)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
