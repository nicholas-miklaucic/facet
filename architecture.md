# Architecture

The repository is constructed to be able to model a wide variety of MACE-family models. It's useful
to think about the different choices one has when creating a model in this vein. That's what this
document does: it lists the choices one has and some thoughts on different options.

(Cool diagram coming eventually.)

First, to go over the blocks we're combining into a model.

We start with an orbit graph: each edge has a distance, each node has a species. Multiple edges can
exist between the same nodes, corresponding to different unit cell shifts. Constructing these graphs
is tremendously expensive for the full MPTrj dataset, so how this is constructed can't change much.
It is currently set up as a 16-nearest-neighbor graph. A cutoff can be used to remove interactions
above some distance.

Then, we have to initialize our node features. The only input we can use here is
composition-level information and global conditioning, if that exists. This creates a species
embedding that can be used in the future.

The next step is message passing. This has two stages: message generation, and message aggregation.
Each edge has features that are generated as a weighted set of spherical harmonics, with weights
given by functions of the distance (expressed using radial basis functions.) These are limited by
equivariance.

As input, message generation can use:
 - The norm of the vector
 - The spherical harmonics applied to the vector
 - Sender/receiver node features
 - Sender/receiver node species

The output can be any kind of irreps.

Message aggregation is some permutation-invariant function that can be applied on any number of
messages, producing a single output. This can use all of the inputs to message generation.

This aggregated message output can be concatenated with the node information, but that needs to be
somehow reduced in dimension to start the next layer. This is done with a self-interaction layer.
The inputs here can be node species, node features, and message information, and it has a target
irreps.

Each of the layers in the network need a readout layer, which converts the given node feature
irreps from the last step into outputs.

That produces one set of outputs per block. Those need to be reduced, although permutation
invariance is not a requirement here. Taking just the last block is equivalent to not having these
readouts, but other models often do a mean across blocks or a sum.

Node-level outputs can be additionally combined with node species information. The standard here is
a species-wise rescaling of energy, but in theory something else could be done.

If the outputs are all scalars, then that opens the door for more flexible readout layers, like
MLPs.

What options do we have for these blocks?

## Species Embedding
This block is basically always a standard embedding layer, with output all scalars. (I cannot even
think of a reasonable interpretation of what a vector or tensor associated with a species would be.)

One improvement to make over much existing literature is the use of a small embedding that gets
projected into the appropriate spaces, instead of having to decide between either having zero
dependence on species or separate weights/values for every species. Having a single species
embedding that is used in different places (projected up to the first node embedding, projected to
the necessary weights for message passing, etc.) avoids overfitting but also allows the element type
to impact things.

## Message Generation
As discussed above, I could easily see having an MLP that produces weights from the RBFs and two
species embeddings. The spherical harmonic tensor product part can't really change, but it is
possible to be flexible with those inputs.

## Message Aggregation
The simplest answer is to do a sum and then divide by some normalizing constant, roughly the average
number of neighbors (beyond which the cutoff is applied.)

In principle, you can do different kinds of attention or gated nonlinearities here. Attention was
unstable when I tried it, but perhaps there's a way of making it work.

Both this and message generation happen 16 times as much as the other layers, so they have
relatively stringent memory and compute requirements. Perhaps it makes sense to make this simple so
the node interactions can be more complex.

## Self Interaction
It remains to be investigated whether including species information here helps at all, or whether
the node features should be a strict upgrade.

This is one of the only major differences between MACE and SevenNet! MACE uses symmetric tensor
contraction to basically compute n, n^2, ..., n^(correlation) up to a max degree and then reduce
back to the desired output. This is enormously expensive, and also makes normalization challenging.
SevenNet uses a linear layer instead (multiplying each group of the same irrep by a matrix), which
is perhaps a little worse but much more parameter efficient.

There's a lot of potential (hah) to split the difference. Some kind of MLP, some kind of gated
nonlinearity, some kind of more limited tensor square.