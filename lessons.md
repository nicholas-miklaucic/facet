# Lessons Learned


## Msgpack seems to work well enough
(Commit 5ecc2037f00575fcec3a591f431c76a7e499e320)

Zarr is around twice as space-efficient when using zip file storage, but 2.5x as slow when reading
in all of the data (1:17 vs. 3:04 for reading in the training data twice in a row, dataset.py
script). Msgpack seems to be the clear speed winner for now. Also, given that read speed, it seems
like data reads aren't bottlenecking much, even without pre-fetching implemented.