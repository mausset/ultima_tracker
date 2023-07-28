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
DETR model and criterion classes.
"""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List

from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .position_encoding import PositionEmebeddingFourierLearned
from .backbone import build_backbone
from .matcher import build_matcher
# from .deformable_transformer_plus import build_deforamble_transformer, pos2posemb
from .deformable_transformer import build_deforamble_transformer
from .qim import build as build_query_interaction_layer
from .deformable_detr import SetCriterion, MLP, sigmoid_focal_loss

from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
import lap


def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb


class TrainingTracker(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.seen = set()
        self.tracks = {}

    def reset(self):
        self.tracks = {}
        self.seen = set()

    def get_active_tracks(self):
        """ Returns the tracks that are active, in order of the labels. """
        tracks_active = []
        for label in sorted(self.seen):
            track = self.tracks[label]
            tracks_active.append(track)
        return torch.stack(tracks_active), torch.tensor(sorted(self.seen))
    
    def update_tracks(self, tracks_new, labels):
        labels = labels.to(torch.long).tolist()
        for label, track in zip(labels, tracks_new):
            self.tracks[label] = track
    
    def init_new_tracks(self, tracks_new, labels):
        labels = labels.to(torch.long).tolist()
        for label, track in zip(labels, tracks_new):
            if label not in self.seen:
                self.tracks[label] = track
                self.seen.add(label)
        
class RuntimeTracker(nn.Module):

    def __init__(self, similarity_threshold, miss_tolerance) -> None:
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.miss_tolerance = miss_tolerance

        self.time_since_seen = {}
        self.tracks = {}
        self.max_id = 0

    def reset(self):
        self.time_since_seen = {}
        self.tracks = {}
        self.max_id = 0
 
    def _init_fresh_tracks(self, proposal_queries):
        for proposal_query in proposal_queries:
            self.tracks[self.max_id] = proposal_query
            self.max_id += 1

    def get_tracks(self):
        if self.max_id == 0:
            return None, None
        track_ids, tracks = zip(*self.tracks.items())
        tracks = torch.stack(tracks).clone() # Make sure not to modify the original tracks, not sure if this is necessary

        return track_ids, tracks

    def _cosine_sim(self, track_queries, proposal_queries):
        track_queries = F.normalize(track_queries, dim=-1)
        proposal_queries = F.normalize(proposal_queries, dim=-1)

        cosine_sim = torch.matmul(track_queries, proposal_queries.T)

        return cosine_sim
    
    def get_assigments(self, proposal_queries, track_queries=None, track_ids=None):
        if track_queries is None:
            self._init_fresh_tracks(proposal_queries)
            return
        
        proposal_queries = proposal_queries.cpu() 
        track_queries = track_queries.cpu()

        cosine_sim = self._cosine_sim(track_queries, proposal_queries)

        # Solve the linear assignment problem using the Hungarian algorithm and get correct matches
        cost, row, col = lap.lapjv(-cosine_sim.numpy(), extend_cost=True, cost_limit=-self.similarity_threshold)

        for identity, r in zip(track_ids, row):
            if r == -1:
                self.time_since_seen[track_ids[identity]] += 1
                continue
            self.time_since_seen[track_ids[identity]] = 0

        for i, proposal, c in enumerate(zip(proposal_queries, col)):
            if c == -1:
                self.tracks[self.max_id] = proposal
                self.time_since_seen[self.max_id] = 0
                col[i] = self.max_id
                self.max_id += 1
            
        return col
    
    def update_tracks(self, track_ids, updated_track_queries):
        for track_id, track in zip(track_ids, updated_track_queries):
            self.tracks[track_id] = track 

    def cull_tracks(self):
        for track_id in list(self.tracks.keys()):
            if self.time_since_seen[track_id] > self.miss_tolerance:
                del self.tracks[track_id]
                del self.time_since_seen[track_id]


class ContrastiveCriterion(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.criterion = SelfSupervisedLoss(NTXentLoss())
        self.binary_cross_entropy = nn.BCEWithLogitsLoss()

    def calculate_cosine_sim_accuracy(self, track_encodings, proposal_features, return_similarities=False):
        track_encodings = F.normalize(track_encodings, dim=-1)
        proposal_features = F.normalize(proposal_features, dim=-1)

        cosine_sim = torch.matmul(track_encodings, proposal_features.T)

        # Solve the linear assignment problem using the Hungarian algorithm and get correct matches
        cost, row, col = lap.lapjv(-cosine_sim.cpu().detach().numpy(), extend_cost=True)
        row = torch.from_numpy(row).cuda()

        correct = (row == torch.arange(row.shape[0]).cuda())

        accuracy = torch.mean(correct.float())

        return accuracy
    
    # Outputs are logits
    def forward_bce(self, outputs, targets):
        loss = self.binary_cross_entropy(outputs, targets)
        acc = torch.mean(((outputs > 0) == targets).float())
        return loss, acc
 
    def forward(self, track_labels, track_queries, proposal_labels, proposal_queries, include_unmatched=False):
        track_labels = track_labels.to(torch.long)
        proposal_labels = proposal_labels.to(torch.long)

        track_labels_list = track_labels.tolist()
        proposal_labels_list = proposal_labels.tolist()

        common_labels = set(track_labels_list) & set(proposal_labels_list)
        track_mask_common = torch.tensor([label in common_labels for label in track_labels_list], dtype=torch.bool)
        proposal_mask_common = torch.tensor([label in common_labels for label in proposal_labels_list], dtype=torch.bool)

        track_labels_common = track_labels[track_mask_common]
        track_queries_common = track_queries[track_mask_common]

        proposal_labels_common = proposal_labels[proposal_mask_common]
        proposal_queries_common = proposal_queries[proposal_mask_common]

        if track_labels_common.shape[0] != 0:
            sorted_track_queries_common, _ = zip(*sorted(zip(track_queries_common, track_labels_common), key=lambda x: x[1]))
            matched_track_queries = torch.stack(sorted_track_queries_common)
        else:
            matched_track_queries = torch.zeros((0, track_queries_common.shape[1]), dtype=track_queries_common.dtype, device=track_queries_common.device)

        if proposal_labels_common.shape[0] != 0:
            sorted_proposal_queries_common, _ = zip(*sorted(zip(proposal_queries_common, proposal_labels_common), key=lambda x: x[1]))
            matched_proposal_queries = torch.stack(sorted_proposal_queries_common)
        else:
            matched_proposal_queries = torch.zeros((0, proposal_queries_common.shape[1]), dtype=proposal_queries_common.dtype, device=proposal_queries_common.device)
        
        if matched_track_queries.shape[0] == 0 or matched_proposal_queries.shape[0] == 0: # Just ignore
            return None, None
        
        assert matched_track_queries.shape == matched_proposal_queries.shape, f"{matched_track_queries.shape} != {matched_proposal_queries.shape}"
            
        loss = self.criterion(matched_track_queries, matched_proposal_queries)
        
        acc = self.calculate_cosine_sim_accuracy(matched_track_queries, matched_proposal_queries)
        
        return loss, acc
 

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MOTR(nn.Module):

    def __init__(self, backbone, transformer, num_feature_levels):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels

        self.proposal_embed = PositionEmebeddingFourierLearned(
            num_pos_feats=hidden_dim//2,
            out_dim=hidden_dim
        )

        self.multi_head_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1)
        self.query_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.predict_fp_tp = MLP(hidden_dim, hidden_dim, 1, 3)
        

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None, proposals=None):
        raise NotImplementedError()
    
    def query_interaction(self, tracks_new, tracks_old):
        x = tracks_new + tracks_old
        q = self.query_weight(x)
        k = self.key_weight(x)
        v = self.value_weight(tracks_new)

        x = self.multi_head_attention(q[:, None], k[:, None], value=v[:, None])[0][:, 0]
        x = x + tracks_new

        x1 = self.layer_norm(x)

        x = self.ffn(x1)
        x = x + x1
        x = self.layer_norm(x)

        return x

    def forward(self, samples: NestedTensor, proposals, tracks=None, proposal_scores=None):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        
        proposals = self.proposal_embed(proposals)
        if proposal_scores is not None:
            proposals = proposals + pos2posemb(proposal_scores, num_pos_feats=proposals.shape[-1])

        query_embed = proposals
        mask = None

        if tracks is not None:
            query_embed = torch.cat([query_embed, tracks], dim=0)
            # mask = torch.zeros((query_embed.shape[0], query_embed.shape[0]), dtype=torch.bool, device=query_embed.device)
            # mask[:proposals.shape[0], proposals.shape[0]:] = True
            # mask[proposals.shape[0]:, :proposals.shape[0]] = True

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embed, src_mask=mask) 

        proposal_queries = hs[-1, 0, :proposals.shape[0]]

        proposal_logits = self.predict_fp_tp(proposal_queries)


        out = {
            'proposal_queries': proposal_queries,
            'proposal_logits': proposal_logits,
            'track_queries': None,
        }

        if tracks is not None:
            track_queries = hs[-1, 0, proposals.shape[0]:]
            out['track_queries'] = track_queries
            out['track_logits'] = self.predict_fp_tp(track_queries)
         
        return out


def build(args):
    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    criterion = ContrastiveCriterion()

    model = MOTR(
        backbone,
        transformer,
        num_feature_levels=args.num_feature_levels,
    )

    return model, criterion, None
