#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import openturns as ot

ot.RandomGenerator.SetSeed(42)
np.random.seed(42)

M = 10000 # Size of the dataset
N = 4  # Dimension of the random vector

# Correlation matrix definition
R = ot.CorrelationMatrix(N)
R[0,1] = 0.3
R[2,3] = -0.3

# Sampling from standard normal
D = ot.Normal([0] * N, [1] * N, R).getSample(M)

D.exportToCSVFile("data/gaussian_sample_01_23.csv", ',')