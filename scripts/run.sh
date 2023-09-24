#!/bin/bash
#pip install -e .
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi
stty cols 130
#stty size | awk '{print $2}'
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
unified_ar $@
