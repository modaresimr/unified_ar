parallel -j3 --jl mylog 'CUDA_VISIBLE_DEVICES=$[{%}-1] python test.py {}' ::: A B C D
