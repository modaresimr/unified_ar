#!/bin/bash

current_date=$(date +"%Y%m%d-%H%M%S")
filename=$(echo "$@" | tr ' ' '_'| tr '-' '_')
echo $filename
save_dir=logs/${filename}_${current_date}
mkdir -p logs


log="${save_dir}_%j.log"

# options="-N 1-1 -J nogpu$filename --gpus=1"
options="-N 1-1 -J '$filename'"


runcmd="./run.sh $@ --output=$save_dir"

do_srun=0

if [ $do_srun == '1' ] ;then
srun $options $runcmd
else
options="$options --output=$log --error=$log "
sbatch $options $runcmd
fi
