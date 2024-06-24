import os

# from beartype.claw import beartype_this_package  # <-- hype comes

# beartype_this_package()  # <-- hype goes

BASE_XLA_FLAGS = """
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_triton_gemm=false
--xla_gpu_simplify_all_fp_conversions
--xla_gpu_enable_async_all_gather=true
--xla_gpu_enable_async_reduce_scatter=true
--xla_gpu_enable_highest_priority_async_stream=true
--xla_gpu_enable_triton_softmax_fusion=false
--xla_gpu_all_reduce_combine_threshold_bytes=33554432
--xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true bash run_pile_multinode.sh"""

os.environ['XLA_FLAGS'] = BASE_XLA_FLAGS.replace('\n', ' ')
