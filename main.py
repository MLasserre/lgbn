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

afn = sys.argv[1]
pfn = sys.argv[2]
tfn = sys.argv[3]
ofn = sys.argv[4]

def prediction(row, cfs, variables):
    evidence = row.drop(['P_res_tp1']).to_dict()
    res = ve.sum_product_ve([], cfs, evidence, variables)
    return res.to_gaussian()[1][0][0]

# Chargement de la structure et des parametres appris par bnlearn
with open(afn, 'r') as myfile:
    arcs = myfile.read()
with open(pfn, 'r') as myfile2:
    parameters = myfile2.read()

parameters = parameters.replace('(', '')
parameters = parameters.replace(')', '')
parameters = parameters[:-2] + parameters[-1]
parameters = parameters.replace(' ', ':')
parameters = parameters.replace('\n', ',')

arcs = ';'.join(arcs.split(sep="\n")[0:-1])

# Chargement des donnees
test = pd.read_csv(tfn, sep=',', header=0)
test['Date'] = pd.to_datetime(test.Date, format='%d/%m/%Y %H:%M:%S')
test = test.set_index('Date')

# Transformation de la chaine en un dictionnaire
parameters = ast.literal_eval(parameters)

# Creation d'un BN "fantome" pour calculer l'abre de jonction
b = gum.fastBN(arcs, 2)
ie = gum.LazyPropagation(b)
print(b)

# Creation des variables
variables = {str(n):cv.ContinuousVariable(n) for n in b.names()}

# Creation des CFs
cfs = []
for v in variables:
    uncond_scope= []
    cond_scope = []
    mu = 0
    Sigma = 0
    B = []

    for p in parameters[v]:
        if p == 'SD':
            Sigma = parameters[v][p]
        elif p == 'Intercept':
            mu = parameters[v][p]
        else:
            cond_scope.append(variables[p])
            B.append(parameters[v][p])
            
    # Le scope est trie par id croissant ainsi que B
    sorted_lists = sorted(zip(cond_scope, B), key=lambda x: x[0])
    cond_scope, B = [[x[i] for x in sorted_lists] for i in range(2)]

    # Les variables sont mises sous la forme de vecteurs/matrices
    mu = np.reshape(mu,(1,1))
    Sigma = np.reshape(Sigma*Sigma, (1,1))

    # La variable associee au noeud est ajoute
    uncond_scope.append(variables[v])
    uncond_scope.sort()
    # Le scope est a nouveau trie (ce qui n'a aucune incidence sur B)
    #scope.sort()             

    # Une fois les parametres recuperes, on cree enfin les CFs
    if not cond_scope:
        cfs.append(cf.CanonicalForm.from_gaussian(uncond_scope, mu, Sigma))
    else:
        B = np.reshape(B, (len(B),1))
        cfs.append(cf.CanonicalForm.from_cond_gaussian(uncond_scope,
                                                       cond_scope,
                                                       mu,
                                                       Sigma, 
                                                       B))

prediction = test.apply(prediction, args=(cfs, variables), axis=1)
test['Prediction'] = prediction

test.to_csv(tfn, index=True, date_format='%d/%m/%Y %H:%M:%S')

# Ecriture de la table
if not Path(ofn).exists():
    with open(ofn, 'w', newline='\n') as o:
        o.write(header)
        o.write('\n')

with open(ofn, 'a') as o:
    o.write(row +'\n')
