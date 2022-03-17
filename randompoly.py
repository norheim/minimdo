import numpy as np
import copy
from functools import reduce
import operator

# RANDOM PROBLEM BUILDING FUNCTIONS:

# Generates an array of size n of small random number between -1 and +1
def smallrand(rng, n=1):
    newn = np.round(rng.uniform(low=-1, high=1),n)
    if newn == 0.: # if we are unlucky because of the rounding
        return smallrand()
    return newn

# Generates a random partition of a set of variables [1,2,3] into [[3,2],[1]]
def generate_partitions(rng, vrs):
    rvrs = copy.copy(list(vrs)) # For immutability
    rng.shuffle(rvrs)
    built_partition = []  # output partition set
    while len(rvrs) >= 3: # 3 is chosen to avoid to many linear/bilinear terms
        ri = rng.integers(2,4) # we wante more than 1 to avoid linear terms
        built_partition.append((smallrand(rng), rvrs[:ri]))
        rvrs = rvrs[ri:] # this is essentially "popping" the element we just added to the partition
    if rvrs:
        built_partition.append((smallrand(rng), rvrs))
    return built_partition

def random_bijective_polynomial(rng, vrs, partition=None):
    partition = partition if partition else generate_partitions(rng, vrs)
    return sum([key*reduce(operator.mul, val) for key, val in partition])+smallrand(rng)

