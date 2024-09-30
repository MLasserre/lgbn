# coding: utf-8

import sys
sys.path.insert(0, "/home/marvin/git_repos/usingOtagrum/sum2019/")

import numpy as np
import canonical_form as cf
import continuous_variable as cv
import openturns as ot

# Building a structure :
# A->B->C->D->E->F->G->H;I->C;E->J;K->H<-L

# Defining continuous variables
A = cv.ContinuousVariable('A')
B = cv.ContinuousVariable('B')
C = cv.ContinuousVariable('C')
D = cv.ContinuousVariable('D')
E = cv.ContinuousVariable('E')
F = cv.ContinuousVariable('F')
G = cv.ContinuousVariable('G')
H = cv.ContinuousVariable('H')
I = cv.ContinuousVariable('I')
J = cv.ContinuousVariable('J')
K = cv.ContinuousVariable('K')
L = cv.ContinuousVariable('L')


# Defining CFs

c_A = cf.CanonicalForm.from_gaussian([A], np.array([[2]]), np.array([[0.002]]))
c_B = cf.CanonicalForm.from_cond_gaussian([B], [A],
                                          np.array([[1]]),
                                          np.array([[0.001]]),
                                          np.array([[-0.2]]))
c_C = cf.CanonicalForm.from_cond_gaussian([C], [B,I],
                                          np.array([[-3]]),
                                          np.array([[0.04]]),
                                          np.array([[2],[-1]]))
c_D = cf.CanonicalForm.from_cond_gaussian([D], [C],
                                          np.array([[0]]),
                                          np.array([[0.005]]),
                                          np.array([[1]]))
c_E = cf.CanonicalForm.from_cond_gaussian([E], [D],
                                          np.array([[-1]]),
                                          np.array([[0.001]]),
                                          np.array([[1]]))
c_F = cf.CanonicalForm.from_cond_gaussian([F], [E],
                                          np.array([[-1]]),
                                          np.array([[0.001]]),
                                          np.array([[-1]]))
c_G = cf.CanonicalForm.from_cond_gaussian([G], [F],
                                          np.array([[-1]]),
                                          np.array([[0.001]]),
                                          np.array([[-2]]))
c_H = cf.CanonicalForm.from_cond_gaussian([H], [G,K,L],
                                          np.array([[-1]]),
                                          np.array([[0.001]]),
                                          np.array([[-0.5],[1],[0.4]]))
c_I = cf.CanonicalForm.from_gaussian([I], np.array([[-1]]), np.array([[2]]))
c_J = cf.CanonicalForm.from_cond_gaussian([J], [E],
                                          np.array([[-1]]),
                                          np.array([[0.001]]),
                                          np.array([[5]]))
c_K = cf.CanonicalForm.from_gaussian([K], np.array([[-0.5]]), np.array([[0.04]]))
c_L = cf.CanonicalForm.from_gaussian([L], np.array([[5]]), np.array([[0.1]]))

# Product of CFs
c_prod = c_A * c_B * c_C * c_D * c_E * c_F * c_G * c_H * c_I * c_J * c_K * c_L

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
D.exportToCSVFile("gbn2.csv")