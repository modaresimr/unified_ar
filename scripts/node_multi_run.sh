#!/bin/bash

mkdir -p logs

# Declaring arrays for datasets, segments, models, and features
datasets=(0 1 2 3 4)
datasets=(3)
segs=(0 1 2 3 4)
segs=(5)
#models=(0 1 2 3 4 5)
# models=(0 1 2 3 4)
models=(1)
feats=(0)
combiner=(0 1)
combiner=(0)
# Looping through each array
for d in "${datasets[@]}"; do
  for s in "${segs[@]}"; do
    for f in "${feats[@]}"; do
      for m in "${models[@]}"; do
for com in ${combiner[@]};do
      # for siz in 1 2 5 15`seq 10 10 60`;do
        #   for ove in 1 2 5 15 `seq 10 10 $s`;do
        for siz in 30; do
          for ove in 5; do
            sp="size=$siz,shift=$ove"

            # Running the commands
#            srun -N 1-1 --gpus=1 cat /etc/hostname

            ./node_single_run.sh -d "$d" -s "$s" -sp "$sp" -f "$f"  -st 0 -m "$m" -c "com" --combiner $com
#	    exit
          done
          done
        done
      done
    done
  done
done
