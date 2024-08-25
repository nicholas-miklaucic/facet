#!/bin/bash
export LD_LIBRARY_PATH=/home/nmiklaucic/nccl_2.21.5-1+cuda11.0_x86_64/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export XLA_FLAGS="
--xla_gpu_enable_latency_hiding_scheduler=true 
--xla_gpu_enable_pipelined_p2p=true
--xla_gpu_enable_triton_softmax_fusion=true
--xla_gpu_triton_gemm_any=True
--xla_gpu_enable_async_collectives=true
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_highest_priority_async_stream=true
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_triton_gemm=false
--xla_gpu_simplify_all_fp_conversions=true
--xla_gpu_enable_async_all_gather=true
--xla_gpu_enable_async_reduce_scatter=true
--xla_gpu_enable_highest_priority_async_stream=true
--xla_gpu_all_reduce_combine_threshold_bytes=33554432
--xla_gpu_graph_level=0 
--xla_gpu_enable_async_all_reduce=true
"
export JAX_TRACEBACK_IN_LOCATIONS_LIMIT=20
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
source secrets.sh
export NEPTUNE_PROJECT="facet"