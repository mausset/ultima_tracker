#!/usr/bin/env bash
for i in $(seq 10 20); do
    threshold=$(echo "scale=2; 1/20 * $i" | bc -l)
    echo $threshold
    ./tools/eval_dance.sh exps/$1/ $threshold
    mv exps/$1/tracker/* TrackEval/data/trackers/mot_challenge/dancetrack-val/ultima/data/
    python ./TrackEval/scripts/run_mot_challenge.py --BENCHMARK dancetrack --SPLIT_TO_EVAL val --TRACKERS_TO_EVAL ultima --METRICS HOTA CLEAR Identity > ./results/results_$threshold.txt
done