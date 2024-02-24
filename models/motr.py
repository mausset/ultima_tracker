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
from torchvision.ops import sigmoid_focal_loss
from torch import nn, Tensor
from typing import List
 
from util.box_ops import generalized_box_iou, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

from util.misc import NestedTensor, inverse_sigmoid

from .position_encoding import PositionEmebeddingFourierLearned
from .backbone import build_backbone

from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
import lap

from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from einops import rearrange

from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder

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
        if len(self.tracks) == 0:
            return None, None

        tracks = torch.stack([torch.stack(track) for track in self.tracks.values()])
        observation_masks = torch.stack([torch.tensor(observation_mask) for observation_mask in self.observation_mask.values()]).bool()

        return tracks.clone(), observation_masks.clone()

class ContrastiveCriterion(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = SelfSupervisedLoss(NTXentLoss())
    
    def bbox_loss(self, predictions, targets):
        """ L1 and GIoU loss for the bounding boxes.
            Expect inputs to be in [x0, y0, x1, y1] format.
        """

        # L1 loss
        loss_l1 = F.l1_loss(predictions, targets, reduction='mean')

        # GIoU loss
        loss_giou = (1 - torch.diag(generalized_box_iou(
            predictions,
            targets
        ))).mean()

        return loss_l1 + loss_giou        
    
    def confidence_accuracy(self, conf_logits, conf_targets):
        conf_targets = conf_targets.bool()
        conf_acc = ((conf_logits > 0) == conf_targets).float().mean()
        positive_acc = ((conf_logits > 0) == conf_targets)[conf_targets].float().mean()
        negative_acc = ((conf_logits > 0) == conf_targets)[~conf_targets].float().mean()
        return conf_acc, positive_acc, negative_acc

    def confidence_loss(self, conf_logits, conf_targets):
        conf_loss = sigmoid_focal_loss(conf_logits, conf_targets, alpha=0.5, gamma=2.0, reduction="mean")
        return conf_loss

    def cosine_sim_accuracy(self, x, y):
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        cosine_sim = torch.matmul(x, y.T).detach().cpu().numpy()

        _, row, _ = lap.lapjv(1 - cosine_sim)

        return (torch.tensor(row).to(x.device) == torch.arange(x.shape[0]).to(x.device)).float().mean()

    def forward(self, predictions, targets):
        loss = self.loss_fn(predictions, ref_emb=targets)

        return loss

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class IterativeRefinement(nn.Module):

    def __init__(self, d_model=256, n_layer=6, position_embed=None) -> None:
        super().__init__()
    
        self.layers = [
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=8,
                norm_first=True,
                dim_feedforward=4*d_model,
                batch_first=True
            ) for _ in range(n_layer)
        ]

        assert position_embed is not None
        self.position_embed = position_embed 
        self.score_embed = PositionEmebeddingFourierLearned(
            num_pos_feats=d_model//2,
            out_dim=d_model,
            decompose_feats=1
        )

        self.confidence_pred = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, boxes, confidence, memory, tgt_key_padding_mask=None):
        x = self.position_embed(boxes) + self.score_embed(confidence)

        for layer in self.layers:
            x = layer(x, memory, tgt_key_padding_mask=tgt_key_padding_mask)
        
        confidence = self.confidence_pred(x)
     
        return x, confidence

class Predictor(nn.Module):

    def __init__(self, stages=3, d_model=256, n_layer=3) -> None:
        super().__init__()
        
        self.predictors = [
            ContinuousTransformerWrapper(
                max_seq_len=25,
                attn_layers=Decoder(
                    dim=d_model,
                    depth=n_layer,
                    heads=8,
                    dim_head=64,
                    rotary_pos_emb=True,
                    ff_mult=4,
                    attn_flash=True,
                    ff_glu=True
                )
            ).cuda()
            for _ in range(stages)
        ]

        #self.mixers = [
        #    ContinuousTransformerWrapper(
        #        max_seq_len=100,
        #        attn_layers=Encoder(
        #            dim=d_model,
        #            depth=n_layer,
        #            heads=8,
        #            dim_head=64,
        #            ff_mult=4,
        #            attn_flash=True,
        #            ff_glu=True
        #        )
        #    ).cuda()
        #    for _ in range(stages-1)
        #]

    def forward(self, x, src_key_padding_mask=None):
        #for predictor, mixer in zip(self.predictors, self.mixers):
        #    x = predictor(x)
        #    x = rearrange(x, 'b n d -> n b d')
        #    x = mixer(x) #, mask=src_key_padding_mask)
        #    x = rearrange(x, 'n b d -> b n d')

        #x = self.predictors[-1](x)  
        for predictor in self.predictors:
            x = predictor(x)

        return x

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
        
        self.position_embed = PositionEmebeddingFourierLearned(
            num_pos_feats=hidden_dim//2,
            out_dim=hidden_dim,
            decompose_feats=2
        )
       
        image_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            norm_first=True,
            dim_feedforward=4*hidden_dim,
            batch_first=True
        )
        self.image_encoder = nn.TransformerEncoder(image_encoder_layer, num_layers=enc_layers)
        
        self.proposal_decoder = IterativeRefinement(
            d_model=hidden_dim,
            n_layer=dec_layers,
            position_embed=self.position_embed
        )

        self.predictor_head = Predictor(
            stages=2,
            d_model=hidden_dim,
            n_layer=3
        )
    
    def image_feature_grid(self, feature_shape):
        h, w = feature_shape

        x = torch.linspace(0, h - 1, h, device='cuda') / h
        y = torch.linspace(0, w - 1, w, device='cuda') / w
        xx, yy = torch.meshgrid(x, y)

        width, height = 1 / w, 1 / h

        return torch.stack((xx, yy, xx+height, yy+width), dim=-1) 
        
    def predictor(self, x, src_key_padding_mask=None):
        return self.predictor_head(x, src_key_padding_mask=src_key_padding_mask)
     
    def forward(self, samples: NestedTensor, proposals, proposal_scores, tgt_key_padding_mask=None):
        features, _ = self.backbone(samples)
        src, _ = features[-1].decompose()
        src = rearrange(src, 'b c h w -> b h w c')

        src = self.image_projection(src) 
        feature_grid = self.image_feature_grid(src.shape[1:3])
        src = src + self.position_embed(feature_grid)

        src = rearrange(src, 'b h w c -> b (h w) c')

        encoded_src = self.image_encoder(src)
        proposal_query, refined_scores = self.proposal_decoder(proposals, proposal_scores, encoded_src, tgt_key_padding_mask=tgt_key_padding_mask)
        
        return proposal_query, refined_scores


def build(args):
    backbone = build_backbone(args)
    criterion = ContrastiveCriterion()

    model = MOTR(
        backbone,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        predictor_layers=args.predictor_layers,
    )

    return model, criterion, None
