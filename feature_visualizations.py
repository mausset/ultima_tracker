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
from torchvision import transforms as T
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

from models.matcher import HungarianMatcher
from torch.utils.data import Dataset, DataLoader
from util.misc import NestedTensor
from util.box_ops import box_cxcywh_to_xyxy

from einops import rearrange
from tsnecuda import TSNE
from torchreid.reid.models import build_model as reid_build_model
from torchreid.reid.utils import FeatureExtractor
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

class ListImgDataset(Dataset):
    def __init__(self, mot_path, vid, img_list, det_db, mot17=False) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.vid = vid
        self.img_list = img_list
        self.det_db = det_db
        self.mot17 = mot17

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1333 #1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.gt_db = self.load_gt_from_file()
    
    def load_gt_from_file(self, default_conf=0.9):
        gt_db = {}
        gt_path = os.path.join(self.mot_path, self.vid, 'gt', 'gt.txt')

        for line in open(gt_path):
            ts, i, *box, _, _ = line.strip().split(',')[:8]
            ts, i = int(ts), int(i)
            if ts not in gt_db:
                gt_db[ts] = []

            l, t, w, h = list(map(float, box))
            gt_db[ts].append([(l + w / 2),
                                (t + h / 2),
                                w,
                                h,
                                default_conf,
                                i])
 
        for t in gt_db.keys():
            gt_db[t] = torch.as_tensor(gt_db[t]).reshape(-1, 6)
            
        return gt_db
    
    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        if self.mot17:
            f_path = f_path.replace('MOT17/', 'MOT17/images/')
        for line in self.det_db[f_path[:-4] + '.txt']:
            l, t, w, h, s = list(map(float, line.split(',')))
            proposals.append([(l + w / 2) / im_w,
                                (t + h / 2) / im_h,
                                w / im_w,
                                h / im_h,
                                s])
        proposals = torch.as_tensor(proposals).reshape(-1, 5)
        
        img_nr = int(os.path.basename(f_path)[:-4])
        gt_boxes = self.gt_db.get(img_nr, None)
        
        return cur_img, proposals, gt_boxes

    def init_img(self, img, proposals, gt_boxes):
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
        return img, ori_img, proposals, gt_boxes

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, proposals, gt_boxes = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, proposals, gt_boxes)


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
    
    def manual_add(self, proposal_queries, track_ids):
        for proposal_query, track_id in zip(proposal_queries, track_ids):
            if track_id not in self.tracks:
                self.tracks[track_id] = []
            self.tracks[track_id].append(proposal_query)
            self.time_since_seen[track_id] = 0
    
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
    def __init__(self, args, model, tracker1, tracker2, vid):
        self.args = args
        self.detr = model
        self.tracker1 = tracker1
        self.tracker2 = tracker2

        self.vid = vid
        self.seq_num = os.path.basename(vid)
        img_list = os.listdir(os.path.join(self.args.mot_path, vid, 'img1'))
        img_list = [os.path.join(vid, 'img1', i) for i in img_list if 'jpg' in i]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        self.predict_path = os.path.join(self.args.output_dir, args.exp_name)
        os.makedirs(self.predict_path, exist_ok=True)

        #self.reid_feature_extractor = reid_build_model(
        #    name='resnet50',
        #    num_classes=751,
        #    pretrained=True,
        #    use_gpu=True
        #).eval()

        self.reid_feature_extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path='/home/rickard/experiments/ultima_tracker/MOTRv2/osnet_ain_x1_0_imagenet.pth',
            device='cuda'
        )


        self.reid_transform = T.Compose([
            T.Resize((256, 128)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

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
    
    def get_reid_features(self, boxes, image) -> torch.Tensor:
        crops = []
        for box in boxes:
            box = box.long()
            crop = image[box[1]:box[3], box[0]:box[2]].float()
            crop = rearrange(crop, 'h w c -> c h w')
            crop = self.reid_transform(crop)
            crops.append(crop)
        crops = torch.stack(crops)
        features = self.reid_feature_extractor(crops)
        return features

    def detect(self, prob_threshold=0.5, area_threshold=100, vis=False, mot17=False, n_frames=200):
        
        with open(os.path.join(self.args.mot_path, self.args.det_db)) as f:
            det_db = json.load(f)

        loader = DataLoader(ListImgDataset(self.args.mot_path, self.vid, self.img_list, det_db, mot17=mot17), 1, num_workers=2)
        matcher = HungarianMatcher()
        for i, data in enumerate(tqdm(loader, leave=False, desc=self.vid)):
            if i >= n_frames:
                break
            cur_img, ori_img, proposals, gt = [d[0] for d in data]
            cur_img, proposals = cur_img.cuda(), proposals.cuda()
            proposals = proposals[proposals[:, -1] > prob_threshold]
            proposal_scores = proposals[:, -1].unsqueeze(-1)
            
            proposals = box_cxcywh_to_xyxy(proposals[:, :-1])

 
            #gt = gt[gt[:, 0] < 1]
            #gt = gt[gt[:, 1] < 1]

            #if gt.shape[0] == 0:
            #    continue

            seq_h, seq_w, _ = ori_img.shape
            
            gt_boxes = gt[:, :4].cuda()
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
            gt_boxes[:, [0, 2]] /= seq_w
            gt_boxes[:, [1, 3]] /= seq_h


            row, col = matcher(gt_boxes, proposals)

            mask = row != -1
            row = row[mask]
            matched = proposals[row]
            matched_scores = proposal_scores[row]
            gt_ids = gt[:, -1][mask].long().tolist()

            matched = torch.clamp(matched, 0, 1)

            if matched.shape[0] == 0:
                continue

            samples = NestedTensor(
                cur_img,
                torch.zeros((seq_h, seq_w), dtype=torch.bool, device=cur_img.device).unsqueeze(0)
            )
        
            gt_queries, _ = self.detr(samples, matched.unsqueeze(0), proposal_scores=matched_scores.unsqueeze(0)) 
            self.tracker1.manual_add(gt_queries.squeeze(0), gt_ids)

            #gt_unnormalized = normalized_to_pixel_coordinates(gt_boxes.clone(), seq_h, seq_w)
            matched_unnormalized = normalized_to_pixel_coordinates(matched.clone(), seq_h, seq_w)
            reid_features = self.get_reid_features(matched_unnormalized, ori_img)
            self.tracker2.manual_add(reid_features, gt_ids)

        key_val = list(self.tracker1.tracks.items())
        key_val = sorted(key_val, key=lambda x: x[0])[:20]

        all_tracks = torch.cat([torch.stack(track) for _, track in key_val])
        ids = torch.cat([torch.full((len(track),), track_id, dtype=torch.long) for track_id, track in key_val])

        all_tracks = all_tracks.cpu().numpy()
        ids = ids.cpu().numpy()

        embedded = TSNE(n_components=2).fit_transform(all_tracks)

        # plot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        #fig.suptitle(self.seq_num[:-4].capitalize() + " " + self.seq_num[-4:].lstrip('0'), y=0.95)
        fig.suptitle(self.seq_num, y=0.95)

        # plot
        sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=ids, palette="muted", edgecolor='none', ax=ax1)

        ax1.text(embedded[:, 0].min(), embedded[:, 1].min(), 'Ultima', horizontalalignment='left')
        ax1.legend([],[], frameon=False)
        ax1.set_xticks([])
        ax1.set_yticks([])

        key_val = list(self.tracker2.tracks.items())
        key_val = sorted(key_val, key=lambda x: x[0])[:20]
        
        all_tracks = torch.cat([torch.stack(track) for _, track in key_val])
        ids = torch.cat([torch.full((len(track),), track_id, dtype=torch.long) for track_id, track in key_val])
        
        all_tracks = all_tracks.cpu().numpy()
        ids = ids.cpu().numpy()

        embedded = TSNE(n_components=2).fit_transform(all_tracks)

        # plot
        sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=ids, palette="muted", edgecolor='none', ax=ax2)

        ax2.text(embedded[:, 0].min(), embedded[:, 1].min(), 'ReID', horizontalalignment='left') #ax1.set_title('Ultima features')
        ax2.legend([],[], frameon=False)
        ax2.set_xticks([])
        ax2.set_yticks([])

        plt.tight_layout()

        self.predict_path = self.predict_path.replace('tracker', 'figures')

        plt.savefig(os.path.join(self.predict_path, self.seq_num + '.png'))
        plt.close()

if __name__ == '__main__':
    mot17 = True

    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 26
    
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
    tracker1 = RuntimeTracker(args.association_threshold, args.miss_tolerance)
    tracker2 = RuntimeTracker(args.association_threshold, args.miss_tolerance)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr.load_state_dict(checkpoint['model'])
    detr.eval()
    detr = detr.cuda()

    # '''for MOT17 submit''' 
    sub_dir = 'MOT17/train'
    seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
    if 'seqmap' in seq_nums:
        seq_nums.remove('seqmap')
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    vids = vids[rank::ws]

    if mot17:
        vids = [x for x in vids if 'SDP' in x]

    with torch.no_grad():
        for vid in tqdm(vids, desc='vid'):
            tracker1.reset()
            tracker2.reset()
            det = Detector(args, model=detr, tracker1=tracker1, tracker2=tracker2, vid=vid)
            det.detect(args.score_threshold, mot17=mot17)
