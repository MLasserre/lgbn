#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import numpy as np
from lgbn.canonical_form import CanonicalForm
from lgbn.continuous_variable import ContinuousVariable
import openturns as ot

# Building a structure :
# A->B->C->D;E->A->C<-E

# Defining continuous variables
A = cv.ContinuousVariable('A')
B = cv.ContinuousVariable('B')
C = cv.ContinuousVariable('C')
D = cv.ContinuousVariable('D')
E = cv.ContinuousVariable('E')

# Defining CFs
c_A = cf.CanonicalForm.from_cond_gaussian([A], [E],
                                          np.array([[2]]),
                                          np.array([[0.002]]),
                                          np.array([[0.1]]))
c_B = cf.CanonicalForm.from_cond_gaussian([B], [A],
                                          np.array([[1]]),
                                          np.array([[0.001]]),
                                          np.array([[-0.2]]))
c_C = cf.CanonicalForm.from_cond_gaussian([C], [A, B, E],
                                          np.array([[-3]]),
                                          np.array([[0.04]]),
                                          np.array([[2],[1],[-1]]))
c_D = cf.CanonicalForm.from_cond_gaussian([D], [C],
                                          np.array([[0]]),
                                          np.array([[0.005]]),
                                          np.array([[1]]))
c_E = cf.CanonicalForm.from_gaussian([E], np.array([[-1]]), np.array([[0.001]]))

# Product of CFs
c_prod = c_A * c_B * c_C * c_D * c_E

# Switching to gaussian parametrization
gaussian = c_prod.to_gaussian()

# Getting mean and covariance matrix
mu = gaussian[1].flatten()
Sigma = ot.CovarianceMatrix(gaussian[0])

M = 10000 # Size of the dataset

# Sampling from normal distribution with same parameters than gaussian
D = ot.Normal(mu, Sigma).getSample(M)

# Switch to rank space
D = (D.rank()+1)/(D.getSize()+2)

D.exportToCSVFile('gbn1.csv')
