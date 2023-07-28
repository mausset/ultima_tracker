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
        pred_accs = []
        
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
            matched_proposal_scores = proposal_scores[col]

            if matched_proposal_boxes.shape[0] == 0:
                print("No matched boxes")
                break

            labels = torch.zeros((matched_proposal_boxes.shape[0], 1), device=matched_proposal_boxes.device)

            # Unmatched proposals. Col is a tensor of indexes of matched proposals
            unmatched_proposal_boxes = torch.cat([
                proposal_boxes[torch.arange(proposal_boxes.shape[0]) != c] for c in col
            ], dim=0)

            unmatched_proposal_scores = torch.cat([
                proposal_scores[torch.arange(proposal_scores.shape[0]) != c] for c in col
            ], dim=0)

            # Sample false positives from unmatched proposals. At least 0.3 of the batch should be false positives
            num_false_positives = unmatched_proposal_boxes.shape[0] #min(unmatched_proposal_boxes.shape[0], int(1 * matched_proposal_boxes.shape[0]))
            false_positive_idx = torch.randperm(unmatched_proposal_boxes.shape[0])[:num_false_positives]

            if false_positive_idx.shape[0] > 0 and len(training_tracker.seen) > 0:
                matched_proposal_boxes = torch.cat([
                    matched_proposal_boxes,
                    unmatched_proposal_boxes[false_positive_idx]
                ], dim=0)

                matched_proposal_scores = torch.cat([
                    matched_proposal_scores,
                    unmatched_proposal_scores[false_positive_idx]
                ], dim=0)

                labels_fp = torch.ones((false_positive_idx.shape[0], 1), device=labels.device)

                labels = torch.cat([
                    labels,
                    labels_fp
                ], dim=0)
            
            samples = NestedTensor(
                img.unsqueeze(0),
                torch.zeros(img.shape[-2:], dtype=torch.bool, device=img.device).unsqueeze(0)
            )

            if len(training_tracker.seen) == 0:
                outputs = model(samples, matched_proposal_boxes, proposal_scores=matched_proposal_scores.unsqueeze(-1))
                training_tracker.init_new_tracks(outputs['proposal_queries'], gt_obj_ids)
                continue
            
            track_queries, track_labels = training_tracker.get_active_tracks()
            outputs = model(samples, matched_proposal_boxes, tracks=track_queries.clone(), proposal_scores=matched_proposal_scores.unsqueeze(-1))
            
            outputs['proposal_queries'] = outputs['proposal_queries'][:gt_obj_ids.shape[0]] # Remove false positives            

            loss, acc = criterion(track_labels, outputs['track_queries'], gt_obj_ids, outputs['proposal_queries'])#, include_unmatched=True)
            loss_bce, pred_acc = criterion.forward_bce(outputs['proposal_logits'], labels)
            


            if loss is not None:
                loss = loss + loss_bce
                losses.append(loss)
                accs.append(acc)
                pred_accs.append(pred_acc)

            updated_track_queries = model.query_interaction(outputs['track_queries'], track_queries)
            training_tracker.update_tracks(updated_track_queries, track_labels)
            training_tracker.init_new_tracks(outputs['proposal_queries'], gt_obj_ids)

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
