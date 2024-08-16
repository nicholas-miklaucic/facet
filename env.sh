#!/bin/bash
export LD_LIBRARY_PATH=/home/nmiklaucic/nccl_2.21.5-1+cuda11.0_x86_64/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_p2p=true"