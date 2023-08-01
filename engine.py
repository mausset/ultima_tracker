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


def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    num_accumulate_batches: int = 1):
    model.train()
    criterion.train()

    training_tracker = TrainingTracker() 

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.0e}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=200, fmt='{value:.2f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=200, fmt='{value:.2f} ({avg:.2f})'))
    metric_logger.add_meter('p_acc', utils.SmoothedValue(window_size=200, fmt='{value:.2f} ({avg:.2f})'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    matcher = HungarianMatcher() # For matching the proposals to the ground truth boxes

    grad_total_norm = 0
    acc_value = 0

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    cnt = 0
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        cnt += 1
        data_dict = data_dict_to_cuda(data_dict, device)
        losses = []
        accs = []
        pred_accs = [1]
        
        training_tracker.reset()
        for i in range(len(data_dict['imgs'])):
            img = data_dict['imgs'][i]
            gt_instances = data_dict['gt_instances'][i]
            proposals = data_dict['proposals'][i]
            gt_boxes = box_cxcywh_to_xyxy(gt_instances.boxes)
            gt_obj_ids = gt_instances.obj_ids
            proposal_boxes = box_cxcywh_to_xyxy(proposals[..., :4])
            proposal_scores = proposals[..., 4]

            row, col = matcher(gt_boxes, proposal_boxes)[0]
            gt_obj_ids = gt_obj_ids[row]

            # Get all indexes of unmatched proposals
            matched_proposal_boxes = torch.clamp(proposal_boxes[col], min=0, max=1)
            matched_proposal_scores = proposal_scores[col].unsqueeze(1)
            
            if matched_proposal_boxes.shape[0] == 0:
                continue

            samples = NestedTensor(
                img.unsqueeze(0),
                torch.zeros(img.shape[-2:], device=img.device).unsqueeze(0)
            )

            proposal_queries = model(samples, proposals=matched_proposal_boxes, proposal_scores=matched_proposal_scores) 

            ids = gt_obj_ids.to(torch.long).tolist()
            training_tracker.add_queries(proposal_queries, ids)

        positive_data, masks, lengths = training_tracker.get_positive_data()
        if positive_data is None:
            continue
        energies = model.energy_function(positive_data, masks)[:, 1:]
        lengths = lengths - 1
        positive_energies = torch.cat(torch.nn.utils.rnn.unpad_sequence(energies, lengths, batch_first=True))
        
        negative_data, masks, lengths = training_tracker.generate_negative_data()
        if negative_data is None:
            continue
        energies = model.energy_function(negative_data, masks)[:, 1:]
        lengths = lengths - 1
        negative_energies = torch.cat(torch.nn.utils.rnn.unpad_sequence(energies, lengths, batch_first=True))

        hard_negative_data, masks, lengths = training_tracker.generate_hard_negative_data()
        if hard_negative_data is not None:
            energies = model.energy_function(hard_negative_data, masks)[:, 1:]
            hard_negative_energies = torch.cat([x[-1].unsqueeze(1) for x in torch.nn.utils.rnn.unpad_sequence(energies, lengths, batch_first=True)])

            negative_energies = torch.cat([negative_energies, hard_negative_energies])
        
        loss, acc = criterion(positive_energies, negative_energies)
        losses.append(loss)
        accs.append(acc)

        # print("iter {} after model".format(cnt-1))

        if len(losses) == 0:
            continue

        loss_value = sum(losses).item() / len(losses)
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = sum(losses) / len(losses) / num_accumulate_batches
        acc_value = sum(accs) / len(accs)
        pred_acc_value = sum(pred_accs) / len(pred_accs)

        loss.backward()
        if cnt % num_accumulate_batches == 0:
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(loss=loss_value)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        metric_logger.update(acc=acc_value, **{'a{}'.format(i): accs[i] for i in range(len(accs))})
        metric_logger.update(p_acc=pred_acc_value)
        # gather the stats from all processes

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
