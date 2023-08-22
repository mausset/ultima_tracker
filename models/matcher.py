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
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import lap
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from models.structures import Instances


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, iou_threshold=0.5):
        super().__init__()
        self.iou_threshold = iou_threshold

    def forward(self, target_bbox, proposal_bbox):
        """ Performs the matching

        Params:
            target_bbox: tensor bounding boxes of the targets, Shape: [batch_size, num_target_boxes, 4]
            proposal_bbox: tensor bounding boxes of the proposals, Shape: [batch_size, num_proposal_boxes, 4]
            boxes are in [x0, y0, x1, y1] format

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            # Compute the IoU
            cost_iou = -box_iou(target_bbox, proposal_bbox)[0]

            # TODO: Look into this
            # # Compute the giou cost betwen boxes
            # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
            #                                  box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = cost_iou.cpu().numpy()

            cost, row, col = lap.lapjv(C, cost_limit=-self.iou_threshold, extend_cost=True)
            return torch.as_tensor(row, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)


def build_matcher(args):
    return HungarianMatcher(args.iou_threshold)
