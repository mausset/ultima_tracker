# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from copy import deepcopy
import json

import os
import argparse
import torchvision.transforms.functional as F
from torch.nn.functional import normalize
import torch
import torch.nn as nn
import lap
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from models import build_model
from util.tool import load_model
from main import get_args_parser
import time

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader
from util.misc import NestedTensor
from util.box_ops import box_cxcywh_to_xyxy

from einops import rearrange, repeat

class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list, det_db) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1333 #1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        for line in self.det_db[f_path[:-4] + '.txt']:
            l, t, w, h, s = list(map(float, line.split(',')))
            proposals.append([(l + w / 2) / im_w,
                                (t + h / 2) / im_h,
                                w / im_w,
                                h / im_h,
                                s])
        return cur_img, torch.as_tensor(proposals).reshape(-1, 5)

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, proposals

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, proposals = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, proposals)


class RuntimeTracker(nn.Module):

    def __init__(self, similarity_threshold, miss_tolerance) -> None:
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.miss_tolerance = miss_tolerance

        self.time_since_seen = {}
        self.tracks = {}
        self.max_id = 1
        self.max_context = 25 - 1

    def reset(self):
        self.time_since_seen = {}
        self.tracks = {}
        self.max_id = 1
 
    def init_fresh_tracks(self, proposal_queries):
        assigned_ids = []
        for proposal_query in proposal_queries:
            assigned_ids.append(self._add_track(proposal_query))
        
        return assigned_ids

    def _add_track(self, proposal_query):
        self.tracks[self.max_id] = [proposal_query]
        self.time_since_seen[self.max_id] = 0
        self.max_id += 1

        return self.max_id - 1
    
    def no_proposals(self):
        for track_id in self.tracks.keys():
            self.tracks[track_id].append(torch.zeros((256,), device=self.tracks[track_id][0].device))
            self.time_since_seen[track_id] += 1
        self.cull_tracks()

    def get_tracks(self):
        if self.max_id == 1:
            return None, None
        track_ids, tracks = zip(*self.tracks.items())
        tracks = [torch.stack(track[-self.max_context:]).clone() for track in tracks]

        return track_ids, tracks
 
    def get_assigments(self, sim_matrix, track_ids, proposal_queries):
        if sim_matrix is None:
            return self.init_fresh_tracks(proposal_queries)
        
        _, row, col = lap.lapjv(1-sim_matrix.cpu().numpy(), extend_cost=True, cost_limit=self.similarity_threshold)

        for identity, r in zip(track_ids, row):
            if r == -1:
                self.time_since_seen[identity] += 1
                self.tracks[identity].append(torch.zeros((256,), device=proposal_queries.device))
            else:
                self.time_since_seen[identity] = 0
                self.tracks[identity].append(proposal_queries[r])
            
        assigned_ids = np.zeros_like(col) - 1
        for i, (proposal, c) in enumerate(zip(proposal_queries, col)):
            if c == -1:
                assigned_ids[i] = self._add_track(proposal)
            else:
                assigned_ids[i] = track_ids[c]
         
        return assigned_ids
    
    def cull_tracks(self):
        for track_id in list(self.tracks.keys()):
            if self.time_since_seen[track_id] > self.miss_tolerance:
                del self.tracks[track_id]
                del self.time_since_seen[track_id]


def normalized_to_pixel_coordinates(normalized_coordinates, height, width):
    """
    Converts normalized coordinates (in the [0, 1] range) to pixel coordinates
    (in the [0, height] range).
    """
    x1 = normalized_coordinates[:, 0].clone() * width
    y1 = normalized_coordinates[:, 1].clone() * height
    x2 = normalized_coordinates[:, 2].clone() * width
    y2 = normalized_coordinates[:, 3].clone() * height
    return torch.stack((x1, y1, x2, y2), dim=-1)

class Detector(object):
    def __init__(self, args, model, tracker, vid):
        self.args = args
        self.detr = model
        self.tracker = tracker

        self.vid = vid
        self.seq_num = os.path.basename(vid)
        img_list = os.listdir(os.path.join(self.args.mot_path, vid, 'img1'))
        img_list = [os.path.join(vid, 'img1', i) for i in img_list if 'jpg' in i]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        self.predict_path = os.path.join(self.args.output_dir, args.exp_name)
        os.makedirs(self.predict_path, exist_ok=True)

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def get_similarity_matrix(self, predictions: torch.tensor, proposals: torch.tensor) -> torch.Tensor: 
        if predictions is None:
            return None

        return torch.matmul(normalize(predictions, dim=-1), normalize(proposals, dim=-1).T)
    
    def get_predictions(self, tracks: list[torch.tensor]) -> torch.Tensor:
        if tracks is None:
            return None

        lengths = torch.tensor([len(track) for track in tracks], device=tracks[0].device)
        padded_tracks = torch.nn.utils.rnn.pad_sequence(tracks, batch_first=True)
        predictions = self.detr.predictor(padded_tracks)
        predictions = torch.stack([p[-1] for p in torch.nn.utils.rnn.unpad_sequence(predictions, lengths, batch_first=True)])
        predictions = self.detr.mixer(predictions)

        return predictions

    def detect(self, prob_threshold=0.5, area_threshold=100, vis=False):
        
        with open(os.path.join(self.args.mot_path, self.args.det_db)) as f:
            det_db = json.load(f)

        loader = DataLoader(ListImgDataset(self.args.mot_path, self.img_list, det_db), 1, num_workers=2)
        lines = []
        for i, data in enumerate(tqdm(loader, leave=False, desc=self.vid)):
            cur_img, ori_img, proposals = [d[0] for d in data]
            cur_img, proposals = cur_img.cuda(), proposals.cuda()
            proposals = proposals[proposals[:, -1] > prob_threshold]
            proposal_scores = proposals[:, -1].unsqueeze(-1)
            
            proposals = box_cxcywh_to_xyxy(proposals[:, :-1])
            proposals = torch.clamp(proposals, 0, 1)

            seq_h, seq_w, _ = ori_img.shape

            samples = NestedTensor(
                cur_img,
                torch.zeros((seq_h, seq_w), dtype=torch.bool, device=cur_img.device).unsqueeze(0)
            )

            proposal_queries, refined_scores = self.detr(samples, proposals.unsqueeze(0), proposal_scores=proposal_scores.unsqueeze(0))
            proposal_queries = proposal_queries.squeeze(0)
            refined_scores = refined_scores.squeeze([0, 2])
             
            proposal_queries = proposal_queries[refined_scores > self.args.pred_score_threshold]
            proposals = proposals[refined_scores > self.args.pred_score_threshold]
            refined_scores = refined_scores[refined_scores > self.args.pred_score_threshold] 

            if proposal_queries.shape[0] == 0:
                self.tracker.no_proposals()
                continue

            track_ids, tracks = self.tracker.get_tracks()
            
            predictions = self.get_predictions(tracks)
            sim_matrix = self.get_similarity_matrix(predictions, proposal_queries)
            identities = self.tracker.get_assigments(sim_matrix, track_ids, proposal_queries) 

            self.tracker.cull_tracks()

            bbox_xyxy = normalized_to_pixel_coordinates(proposals, seq_h, seq_w)
            save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
            for xyxy, proposal_id in zip(bbox_xyxy, identities):
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                lines.append(save_format.format(frame=i + 1, id=proposal_id, x1=x1, y1=y1, w=w, h=h))
        
        with open(os.path.join(self.predict_path, f'{self.seq_num}.txt'), 'w') as f:
            f.writelines(lines)

if __name__ == '__main__':
    
    torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--score_threshold', default=0.05, type=float)
    parser.add_argument('--pred_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=24, type=int)
    parser.add_argument('--association_threshold', default=0.5, type=float)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    detr, _, _ = build_model(args)
    tracker = RuntimeTracker(args.association_threshold, args.miss_tolerance)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr.load_state_dict(checkpoint['model'])
    detr.eval()
    detr = detr.cuda()

    # '''for MOT17 submit''' 
    sub_dir = 'DanceTrack/test'
    seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
    if 'seqmap' in seq_nums:
        seq_nums.remove('seqmap')
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    vids = vids[rank::ws]

    with torch.no_grad():
        for vid in tqdm(vids, desc='vid'):
            tracker.reset()
            det = Detector(args, model=detr, tracker=tracker, vid=vid)
            det.detect(args.score_threshold)
