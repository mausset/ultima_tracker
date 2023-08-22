./tools/eval_dance.sh exps/$1/ || { echo 'inference failed' ; exit 1; }
cp exps/$1/tracker/* TrackEval/data/trackers/mot_challenge/dancetrack-val/ultima/data/
python ./TrackEval/scripts/run_mot_challenge.py --BENCHMARK dancetrack --SPLIT_TO_EVAL val --TRACKERS_TO_EVAL ultima --METRICS HOTA CLEAR > results.txt