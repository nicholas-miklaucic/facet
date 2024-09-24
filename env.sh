#!/bin/bash
export LD_LIBRARY_PATH=/home/nmiklaucic/nccl_2.21.5-1+cuda11.0_x86_64/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export XLA_FLAGS="
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_triton_gemm=true
--xla_gpu_triton_gemm_any=True
--xla_gpu_enable_triton_softmax_fusion=true
--xla_gpu_graph_level=0 
"
export JAX_TRACEBACK_IN_LOCATIONS_LIMIT=20
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
source secrets.sh
export NEPTUNE_PROJECT="facet/energy-beta"
