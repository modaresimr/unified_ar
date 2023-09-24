#!/bin/bash

current_date=$(date +"%Y%m%d-%H%M%S")
filename=$(echo "$@" | tr ' ' '_')
save_dir=logs/$filename_$current_date
mkdir -p $save_dir


log="$save_dir/out_%j.log"

options="-N 1-1 -J $name --gpus=1"

runcmd="./run.sh $@ --output=$save_dir"
echo srun $options $runcmd
options="$options --output=$log --error=$log "
sbatch $options $runcmd
