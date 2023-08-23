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
        self.max_id = 0
        self.max_context = 25 - 1

    def reset(self):
        self.time_since_seen = {}
        self.tracks = {}
        self.max_id = 0
 
    def _init_fresh_tracks(self, proposal_queries):
        assigned_ids = []
        for proposal_query in proposal_queries:
            self.tracks[self.max_id] = [proposal_query]
            self.time_since_seen[self.max_id] = 0
            assigned_ids.append(self.max_id)
            self.max_id += 1
        
        return assigned_ids

    def get_tracks(self):
        if self.max_id == 0:
            return None, None
        track_ids, tracks = zip(*self.tracks.items())
        tracks = [torch.stack(track[-self.max_context:]).clone() for track in tracks]

        return track_ids, tracks
 
    def get_assigments(self, sim_matrix, track_ids, proposal_queries):
        if len(self.tracks) == 0:
            return self._init_fresh_tracks(proposal_queries)
        
        # Solve the linear assignment problem using the Hungarian algorithm and get correct matches
        cost, row, col = lap.lapjv(sim_matrix.numpy(), extend_cost=True, cost_limit=self.similarity_threshold)

        for identity, r in zip(track_ids, row):
            if r == -1:
                self.tracks[identity].append(torch.zeros((256,), device=proposal_queries.device))
                self.time_since_seen[identity] += 1
            else:
                self.time_since_seen[identity] = 0
                self.tracks[identity].append(proposal_queries[r])

        for i, (proposal, c) in enumerate(zip(proposal_queries, col)):
            if c == -1:
                self.tracks[self.max_id] = [proposal]
                self.time_since_seen[self.max_id] = 0
                col[i] = self.max_id 
                self.max_id += 1
            else:
                col[i] = track_ids[c]
            
        return col # TODO: This is wrong.
    
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

    def get_similarity_matrix(self, tracks: list[torch.tensor], proposals: torch.tensor) -> torch.Tensor:
        padded_tracks = torch.nn.utils.rnn.pad_sequence(tracks, batch_first=True)
        predictions = self.detr.predictor(padded_tracks)

        
        return torch.matmul(normalize(predictions, dim=-1), normalize(proposals, dim=-1).T)

    def detect(self, prob_threshold=0.5, area_threshold=100, vis=False):
        total_dts = 0
        total_occlusion_dts = 0

        track_instances = None
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

            seq_h, seq_w, _ = ori_img.shape
 
            samples = NestedTensor(
                cur_img,
                torch.zeros((seq_h, seq_w), dtype=torch.bool, device=cur_img.device).unsqueeze(0)
            )

            proposal_queries = self.detr(samples, proposals.unsqueeze(0), proposal_scores=proposal_scores).squeeze(0)
                         
            track_ids, tracks = self.tracker.get_tracks()
            if tracks is None:
                identities = self.tracker._init_fresh_tracks(proposal_queries)
            else:
                sim_matrix = self.get_similarity_matrix(tracks, proposal_queries)
                identities = self.tracker.get_assigments(sim_matrix, track_ids, proposal_queries)

            self.tracker.cull_tracks()

            bbox_xyxy = normalized_to_pixel_coordinates(proposals, seq_h, seq_w)
            save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
            for xyxy, proposal_id in zip(bbox_xyxy, identities):
                if proposal_id < 0 or proposal_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                lines.append(save_format.format(frame=i + 1, id=proposal_id, x1=x1, y1=y1, w=w, h=h))
        
        with open(os.path.join(self.predict_path, f'{self.seq_num}.txt'), 'w') as f:
            f.writelines(lines)

if __name__ == '__main__':
    
    torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    detr, _, _ = build_model(args)
    # detr.track_embed.score_thr = args.update_score_threshold
    tracker = RuntimeTracker(0.31, args.miss_tolerance)
    # detr.track_base = RuntimeTracker(args.score_threshold, args.miss_tolerance)
    checkpoint = torch.load(args.resume, map_location='cpu')
    # detr = load_model(detr, args.resume)
    detr.load_state_dict(checkpoint['model'])
    detr.eval()
    detr = detr.cuda()

    # '''for MOT17 submit''' 
    sub_dir = 'DanceTrack/val'
    seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
    if 'seqmap' in seq_nums:
        seq_nums.remove('seqmap')
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    vids = vids[rank::ws]

    with torch.no_grad():
        for vid in tqdm(vids, desc='vid'):
            det = Detector(args, model=detr, tracker=tracker, vid=vid)
            det.detect(args.score_threshold)
