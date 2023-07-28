./tools/eval_dance.sh exps/motrv2/$1/
cp exps/motrv2/$1/tracker/* TrackEval/data/trackers/mot_challenge/dancetrack-val/ultima/data/
python ./TrackEval/scripts/run_mot_challenge.py --BENCHMARK dancetrack --SPLIT_TO_EVAL val --TRACKERS_TO_EVAL ultima --METRICS HOTA CLEAR