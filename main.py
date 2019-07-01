# coding: utf-8

import numpy as np
import pyAgrum as gum
import canonical_form as cf
import variable_elimination as ve
import continuous_variable as cv
import pandas as pd
import ast  # Permet la creation d'un dico a partir d'un string
import sys
import os
from pathlib import Path
import sklearn.metrics as metrics

# Creation des variables
A = cv.ContinuousVariable('A')
B = cv.ContinuousVariable('B')
C = cv.ContinuousVariable('C')
D = cv.ContinuousVariable('D')

# Creation des CFs
c_A = 
c_B =
c_C =
c_D = cf.from_cond_gaussian([D], [C], 


for v in variables:
    uncond_scope= []
    cond_scope = []
    mu = 0
    # Sigma = 0
    # B = []

    # for p in parameters[v]:
        # if p == 'SD':
            # Sigma = parameters[v][p]
        # elif p == 'Intercept':
            # mu = parameters[v][p]
        # else:
            # cond_scope.append(variables[p])
            # B.append(parameters[v][p])
            
    # # Le scope est trie par id croissant ainsi que B
    # sorted_lists = sorted(zip(cond_scope, B), key=lambda x: x[0])
    # cond_scope, B = [[x[i] for x in sorted_lists] for i in range(2)]

    # # Les variables sont mises sous la forme de vecteurs/matrices
    # mu = np.reshape(mu,(1,1))
    # Sigma = np.reshape(Sigma*Sigma, (1,1))

    # # La variable associee au noeud est ajoute
    # uncond_scope.append(variables[v])
    # uncond_scope.sort()
    # # Le scope est a nouveau trie (ce qui n'a aucune incidence sur B)
    # #scope.sort()             

    # # Une fois les parametres recuperes, on cree enfin les CFs
    # if not cond_scope:
        # cfs.append(cf.CanonicalForm.from_gaussian(uncond_scope, mu, Sigma))
    # else:
        # B = np.reshape(B, (len(B),1))
        # cfs.append(cf.CanonicalForm.from_cond_gaussian(uncond_scope,
                                                       # cond_scope,
                                                       # mu,
                                                       # Sigma, 
                                                       # B))
