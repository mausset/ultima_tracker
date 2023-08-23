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

from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, Summer
from einops import rearrange, repeat

class TrainingTracker(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.tracks = {}
        self.observation_mask = {}
        self.time_step = 0

    def reset(self):
        self.tracks = {}
        self.observation_mask = {}
        self.time_step = 0
 
    def update(self, queries, labels):
        for query, label in zip(queries, labels):
            new = False
            if label not in self.tracks:
                self.tracks[label] = [torch.zeros_like(query)] * self.time_step
                self.observation_mask[label] = [0] * self.time_step
                new = True
            self.tracks[label].append(query)
            self.observation_mask[label].append(0 if new else 1) # Ignore the first observation of a track
        
        not_seen = set(self.tracks.keys()) - set(labels) 
        for label in not_seen:
            self.tracks[label].append(torch.zeros((256,)).to(queries.device))
            self.observation_mask[label].append(0)
        
        self.time_step += 1
    
    def get_tracks(self): 
        tracks = torch.stack([torch.stack(track) for track in self.tracks.values()])
        observation_masks = torch.stack([torch.tensor(observation_mask) for observation_mask in self.observation_mask.values()]).bool()

        return tracks.clone(), observation_masks.clone()

class ContrastiveCriterion(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = SelfSupervisedLoss(NTXentLoss())
    
    def cosine_sim_accuracy(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        cosine_sim = torch.matmul(x, y.T)

        _, row, _ = lap.lapjv(1 - cosine_sim.cpu().numpy())

        return (torch.tensor(row).to(x.device) == torch.arange(x.shape[0]).to(x.device)).float().mean()

    def forward(self, predictions, targets):
        loss = self.loss_fn(predictions, ref_embed=targets)
        acc = self.cosine_sim_accuracy(predictions, targets)

        return loss, acc

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MOTR(nn.Module):

    def __init__(self, backbone, enc_layers=6, dec_layers=6, predictor_layers=6, hidden_dim=256):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            hidden_dim: the output dimension of the backbone
        """
        super().__init__()
        self.backbone = backbone
        
        self.query_pos_enc = Summer(PositionalEncoding1D(hidden_dim))
        
        self.image_projection = nn.Linear(2048, hidden_dim)        
        self.score_embedding = nn.Linear(1, hidden_dim)

        self.position_embed = PositionEmebeddingFourierLearned(
            num_pos_feats=hidden_dim//2,
            out_dim=hidden_dim
        )
        
        image_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            norm_first=True,
            dim_feedforward=4*hidden_dim,
            batch_first=True
        )
        self.image_encoder = nn.TransformerEncoder(image_encoder_layer, num_layers=enc_layers)
        
        proposal_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            norm_first=True,
            dim_feedforward=4*hidden_dim,
            batch_first=True
        )
        self.proposal_decoder = nn.TransformerDecoder(proposal_decoder_layer, num_layers=dec_layers)
        
        predictor_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            norm_first=True, 
            dim_feedforward=4*hidden_dim, 
            batch_first=True
        )
        self.predictor_encoder = nn.TransformerEncoder(predictor_encoder_layer, num_layers=predictor_layers)
    
    def image_feature_grid(self, feature_shape):
        h, w = feature_shape

        x = torch.linspace(0, h - 1, h, device='cuda') / h
        y = torch.linspace(0, w - 1, w, device='cuda') / w
        xx, yy = torch.meshgrid(x, y)

        width, height = 1 / w, 1 / h

        return torch.stack((xx, yy, xx+height, yy+width), dim=-1)
        
    def predictor(self, x):
        causal_mask = torch.triu(torch.ones((x.shape[1], x.shape[1])), diagonal=1).to(x.device).bool() 
        x = self.query_pos_enc(x)
        x = self.predictor_encoder(x, mask=causal_mask, is_causal=True)
        return x
    
    def forward(self, samples: NestedTensor, proposals, proposal_scores, tgt_key_padding_mask=None):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        src = rearrange(src, 'b c h w -> b h w c')

        src = self.image_projection(src) 
        feature_grid = self.image_feature_grid(src.shape[1:3])
        src = src + self.position_embed(feature_grid)

        src = rearrange(src, 'b h w c -> b (h w) c')
                 
        proposal_query = self.position_embed(proposals)
        if proposal_scores is not None:
            proposal_query = proposal_query + self.score_embedding(proposal_scores)

        encoded_src = self.image_encoder(src)
        proposal_query = self.proposal_decoder(proposal_query, encoded_src, tgt_key_padding_mask=tgt_key_padding_mask)
        
        return proposal_query


def build(args):
    backbone = build_backbone(args)
    criterion = ContrastiveCriterion()

    model = MOTR(
        backbone,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        predictor_layers=args.energy_layers,
    )

    return model, criterion, None
