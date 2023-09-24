#!/bin/bash

mkdir -p logs

dataset=0 1 2 3 4
dataset=0
seg=0 1 2
seg=0
models=0 1 2 3 4 5
feat=0
for d in $dataset
for s in $seg;do
for f in $feat;do
for m in $models;do

# for siz in 1 2 5 15`seq 10 10 60`;do
#   for ove in 1 2 5 15 `seq 10 10 $s`;do
for siz in 30;do
  for ove in 5;do
  	sp="size=$siz,shift=$ove"

    srun -N 1-1 --gpus=1 cat /etc/hostname
    ./node_single_run.sh -c comment -d $d -s $s -sp $sp -f $f  -st 0 -m $m

done
done
done
done
done
