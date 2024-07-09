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
kernel_init=flax.linen.initializers.normal(
    # stddev=jnp.sqrt(alpha) ** (1.0 - gradient_normalization)
    stddev=1
),
```

and a corresponding change in the output