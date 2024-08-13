import numpy as np
import copy
from functools import reduce
import operator

# RANDOM PROBLEM BUILDING FUNCTIONS:

# Generates an array of size n of small random number between -1 and +1
def smallrand(rng, n=1, skew=None):
    drawrandomvar = rng.uniform(low=-1, high=1)
    if skew is not None:
        drawrandomvar = np.random.choice([drawrandomvar, 0], 
                                         p=[1-skew, skew])
    newn = np.round(drawrandomvar, n)
    if newn == 0. and skew is None: # if we are unlucky because of the rounding
        return smallrand(rng)
    return newn

# Generates a random partition of a set of variables [1,2,3] into [[3,2],[1]]
def generate_partitions(rng, lenvrs, xval=None):
    rvrs = np.arange(lenvrs) # For immutability
    rng.shuffle(rvrs)
    built_partition = []  # output partition set
    while len(rvrs) >= 3: # 3 is chosen to avoid too many linear/bilinear terms
        ri = rng.integers(1,4) # we wante more than 1 to avoid linear terms
        built_partition.append((smallrand(rng), rvrs[:ri]))
        rvrs = rvrs[ri:] # this is essentially "popping" the element we just added to the partition
    if len(rvrs)>=1:
        built_partition.append((smallrand(rng), rvrs))
    if len(built_partition) >= 2 and xval is not None:
        remainder = sum([key*reduce(operator.mul, xval[val], 1) for key, val in built_partition[:-1]])
        last_item = built_partition[-1][1]
        remainder = -remainder / reduce(operator.mul, map(lambda x: xval[x], last_item), 1)
        built_partition[-1] = (np.round(remainder,2), last_item)
    return built_partition

def random_bijective_polynomial(rng, vrs, partition=None, xval=None):
    vrs = list(vrs)
    partition = partition if partition else generate_partitions(rng, len(vrs), xval)
    if xval is not None:
        remainder = -sum([key*reduce(operator.mul, xval[val], 1) for key, val in partition])
        remainder = np.round(remainder, 2)
    else:
        remainder = smallrand(rng, skew=0.5) # make the term 0 50% of the time
    return sum([key*reduce(operator.mul, map(lambda x: vrs[x], val), 1) for key, val in partition])+remainder

