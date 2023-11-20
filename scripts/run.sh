#!/bin/bash
#pip install -e .
source ~/.bashrc
conda activate tfn4
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

stty cols 130
#stty size | awk '{print $2}'
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
cd ..
#unified_ar $@
#pip3 install -e .
python3 -m unified_ar $@
#python --version


#python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
