#!/usr/bin/env bash
./tools/eval_dance.sh exps/$1/ $2 $3 $4
cp exps/$1/tracker/* TrackEval/data/trackers/mot_challenge/$3-$4/ultima/data/
rm exps/$1/tracker/*
python ./TrackEval/scripts/run_mot_challenge.py --BENCHMARK $3 --SPLIT_TO_EVAL $4 --TRACKERS_TO_EVAL ultima --METRICS HOTA CLEAR Identity > ./results/results_$2.txt