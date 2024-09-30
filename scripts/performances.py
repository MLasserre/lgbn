#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import openturns as ot
import os

ot.RandomGenerator.SetSeed(42)
np.random.seed(42)

M = 10000 # Size of the dataset
N = 4  # Dimension of the random vector

# Correlation matrix definition
R = ot.CorrelationMatrix(N)
R[0,1] = 0.3
R[2,3] = -0.3

# Sampling from standard normal
data = ot.Normal([0] * N, [1] * N, R).getSample(M)
names = data.getDescription()

data = np.array(data)

n_test = 20
sizes = np.logspace(1, 4, n_test, dtype=int)

for size in sizes:
    sample = data[np.random.randint(0,len(data), size=size)]
    sample = ot.Sample(sample)
    sample.setDescription(names)
    sample.exportToCSVFile("data/multisample/gaussian_sample_01_23_n"+str(size), ',')
    
    
    