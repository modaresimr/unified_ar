#!/bin/bash

current_date=$(date +"%Y%m%d-%H%M%S")
filename=$(echo "$@" | tr ' ' '_'| tr '-' '_')
echo $filename
save_dir=${filename}_${current_date}
slurm="logs/%j_${save_dir}.log"
mkdir -p logs



# options="-N 1-1 -J nogpu$filename --gpus=1"
options="-N 1-1 -J $filename --gpus=1"


runcmd="./run.sh $@ --output=logs/$save_dir"

do_srun=1

if [ $do_srun == '1' ] ;then
srun -w lipn-rtx3 $options $runcmd
else
options="$options --output=$slurm --error=$slurm "
sbatch $options $runcmd
fi
