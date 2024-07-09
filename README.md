# CDV

A reimplementation of CDVAE, planning to add more.

Questions:
- Is it normal for every atom to have the max number of neighbors (20)?


Things to do:

- Implement full multigraph stuff in graph processing
- Fudge incoming indices
- Implement decoder


## Hacks

The monkey patching I've done to make things work:

e3nn mlp flax needs 

```py
kernel_init=flax.linen.initializers.variance_scaling(
    scale=2.0,
    mode='fan_in',
    distribution='truncated_normal'
    # stddev=scale
    # stddev=1
),
```

and a corresponding change in the output


## Improvements

### Normalization
LayerNorm needs to be adapted to E(3) code: otherwise, it's quite challenging to normalize across
different architectures. The existing code is far too brittle.

### Cost Analysis
I need to do a serious analysis of how the parameters should best be utilized.

### Node Embeddings
Instead of using 72 individual sets of matrices, have a smaller set of weight matrices that are
combined based on the embedding of each node.

### Global Conditioning
How to incorporate into the network? Attach to node embeddings?