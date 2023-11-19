#!/usr/bin/env bash
for i in $(seq 4 20); do
    threshold=$(echo "scale=2; 1/20 * $i" | bc -l)
    echo $threshold
    ./tools/eval_dance.sh exps/$1/ $threshold $2 $3
    mv exps/$1/tracker/* TrackEval/data/trackers/mot_challenge/$2-$3/ultima/data/
    python ./TrackEval/scripts/run_mot_challenge.py --BENCHMARK $2 --SPLIT_TO_EVAL $3 --TRACKERS_TO_EVAL ultima --METRICS HOTA CLEAR Identity > ./results/results_$threshold.txt
done