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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TrainingTracker(nn.Module):
    
    def __init__(self, max_length=5) -> None:
        super().__init__()
        self.tracks = {}

    def reset(self):
        self.tracks = {}
 
    def add_queries(self, queries, labels):
        for query, label in zip(queries, labels):
            if label not in self.tracks:
                self.tracks[label] = []
            self.tracks[label].append(query)
    
    def get_positive_data(self):
        positive_data = []
        for label, queries in self.tracks.items():
            if len(queries) < 2:
                continue
            positive_data.append(torch.stack(queries))
        
        if len(positive_data) == 0:
            return None, None, None
        
        padded_seqs = torch.nn.utils.rnn.pad_sequence(positive_data, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence([torch.ones_like(queries[:, 0]) for queries in positive_data], batch_first=True)
        lengths = torch.tensor([len(queries) for queries in positive_data])
            
        return padded_seqs, masks, lengths

    def all_queries(self):
        data = []
        lengths = []
        for label, queries in self.tracks.items():
            data.append(torch.stack(queries))
            lengths.append(len(queries))
            
        return torch.cat(data), torch.tensor(lengths)
     
    # NOTE: There are many ways to generate negative data.
    # Which is optimal remains to be seen.
    # For now, just randomly construct by sampling without replacement
    def generate_negative_data(self):
        positive_data, lengths = self.all_queries()
        negative_data = []
        total_length = positive_data.shape[0]
        for length in lengths:
            sampled_indexes = torch.randperm(total_length)[:length]
            negative_data.append(positive_data[sampled_indexes])

        padded_seqs = torch.nn.utils.rnn.pad_sequence(negative_data, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence([torch.ones_like(queries[:, 0]) for queries in negative_data], batch_first=True) 
        return padded_seqs, masks, lengths

    # Generate negatives by sampling in queries from other tracks
    def generate_hard_negative_data(self):
        negative_data = []
        lengths = []
        if len(self.tracks) < 2:
            return None, None, None
        for label, tmp_queries in self.tracks.items():
            if len(tmp_queries) < 2:
                    continue
            for _ in range(5):
                lengths.append(len(tmp_queries))

                queries = torch.stack(tmp_queries).clone()
                # Switch one of the queries with a query from another track
                idx = torch.randperm(len(self.tracks))
                sampled_label = label
                while sampled_label == label:
                    sampled_label = list(self.tracks.keys())[idx[0]]
                    idx = idx[1:]

                sampled_query = self.tracks[sampled_label][np.random.randint(len(self.tracks[sampled_label]))]
                sampled_index = np.random.randint(queries.shape[0])
                queries[sampled_index] = sampled_query

                negative_data.append(queries)

        if len(negative_data) == 0:
            return None, None, None
    
        padded_seqs = torch.nn.utils.rnn.pad_sequence(negative_data, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence([torch.ones_like(queries[:, 0]) for queries in negative_data], batch_first=True)
        lengths = torch.tensor(lengths)

        return padded_seqs, masks, lengths






        
            
        
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

    def forward(self, positive_energies, negative_energies):
        assert positive_energies is not None or negative_energies is not None
        positive_targets = torch.zeros((positive_energies.shape[0], 1), dtype=torch.float, device=positive_energies.device)
        negative_targets = torch.ones((negative_energies.shape[0], 1), dtype=torch.float, device=negative_energies.device)
        targets = torch.cat([positive_targets, negative_targets])
        all_energies = torch.cat([positive_energies, negative_energies])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(all_energies, targets)
        acc = torch.mean(((all_energies > 0) == targets).float())
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

        self.positional_encoding = PositionalEncoding(hidden_dim, 0,  10)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            norm_first=True, 
            dim_feedforward=4*hidden_dim, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.projector = torch.nn.Linear(hidden_dim, 1, bias=False)


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
    
    def energy_function(self, x, padding_mask):
        causal_mask = torch.triu(torch.ones((x.shape[1], x.shape[1])), diagonal=1).to(x.device)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask=causal_mask, is_causal=True, src_key_padding_mask=padding_mask)
        x = self.projector(x)
        return x
    
    def forward(self, samples: NestedTensor, proposals, proposal_scores=None):
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

        
        proposal_query = self.proposal_embed(proposals)
        if proposal_scores is not None:
            proposal_query = proposal_query + pos2posemb(proposal_scores, num_pos_feats=proposal_query.shape[-1])
        
        reference_points = proposals

        mask = None

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, proposal_query, ref_pts=reference_points, src_mask=mask) 

        proposal_queries = hs[-1, 0, :proposals.shape[0]]

         
        return proposal_queries


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
