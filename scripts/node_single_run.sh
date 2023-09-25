#!/bin/bash

current_date=$(date +"%Y%m%d-%H%M%S")
filename=$(echo "$@" | tr ' ' '_'| tr '-' '_')
echo $filename
save_dir=logs/$filename_$current_date
mkdir -p $save_dir


log="$save_dir/out_%j.log"

options="-N 1-1 -J $name --gpus=1"

runcmd="./run.sh $@ --output=$save_dir"

do_srun=1

if [ $do_srun == '1' ] ;then
 srun $options $runcmd
else
options="$options --output=$log --error=$log "
sbatch $options $runcmd
fi
