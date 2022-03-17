#!/bin/bash
module load anaconda
source activate paddle
module load cuda/11.2
module load cudnn/8.1.0.77_CUDA11.2
module load nccl/2.9.6-1_cuda11.2

python3 -m paddle.distributed.launch ./main.py > result.out