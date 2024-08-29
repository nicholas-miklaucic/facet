# CDV
Working repository for diffusion and prediction models using MACE.

## To Do 
- Maybe give a LR-free method another whirl?
- Test state prediction needs to be parallelized
- Only do tensor products we actually use!
- Get a good baseline for future hyperparameter experimentation



## Ideas
### Prediction Gradients
MACE currently predicts forces and stress by differentiating the energy prediction function
through the positions of the nodes. Is this a better way of doing diffusion?

### Lattice Prediction
The edge images mean that gradients are computable w.r.t the lattice, especially if fractional coordinates are used. Is this better than predicting the lattice modification?

### Constrained Diffusion
What if Doob's h-transform is used to predict atom type by letting the atomic embeddings vary continuously?

### Parameter Allocation
What distribution of irreps leads to the best performance?
What internal representation should be used when doing the linear up and down projections?

### Tensor Products
There are a few different ways of doing tensor products to update node representations. 

We can take tensor products only within a single irrep, or we can take products between all allowed irreps from the input and output. Existing literature doesn't have the off-diagonal entries often, but this echoes the next point...

It seems to me that using lower-rank linear layers could have a lot of potential. The most expensive part of the model is the node update, but it may be better to trade off fidelity there for more ability to let edges communicate.

### Edge Updates
Edge updates can in theory depend on 

- the hidden states of the sender/reciever nodes
- the species of the sender/receiver nodes
- the edge distance

How to best combine this information?

For example, right now, edges are gated using the radial embedding, independent of the actual data. Perhaps this should change?

Alternatively, could dot product attention be made stable?