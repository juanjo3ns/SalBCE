import numpy as np
import torch
import collections

def listtoOrderedDict(weights):
	return collections.OrderedDict(weights)

def adaptWeights(ordict):
	weights = list(ordict)
	param = [[2,'BN_1', torch.rand(1,64)],
			[5,'BN_1', torch.rand(1,64)],
			[2,'BN_1', torch.rand(1,128)],
			[2,'BN_1', torch.rand(1,128)],
			[2,'BN_1', torch.rand(1,128)],
			[2,'BN_1', torch.rand(1,128)],
			[2,'BN_1', torch.rand(1,128)],
			[2,'BN_1', torch.rand(1,128)],
			[2,'BN_1', torch.rand(1,128)],
			[2,'BN_1', torch.rand(1,128)],
			[2,'BN_1', torch.rand(1,128)]]

	for p in param:
		weights.insert(p[0],(p[1],p[2]))

	return listtoOrderedDict(weights)
