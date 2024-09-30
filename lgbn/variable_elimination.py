# coding: utf-8

import numpy as np
import canonical_form as cf
import continuous_variable as cv


def sum_product_ve(elimination_order, cf_set, evidence, variables):
    cf_set_copy = cf_set[:]

    if evidence:
        for i,c in enumerate(cf_set_copy):
            lvn = [str(s) for s in c.get_scope().get_v()]
            for vn in lvn:
                if vn in evidence.keys():
                    cf_set_copy[i] = cf_set_copy[i].reduce([variables[vn]],
                                                       np.array([[evidence[vn]]]))
    
    if elimination_order:
        for v in elimination_order:
            cf_set_copy = sum_product_eliminate_var(cf_set_copy, variables[v])
    cf_set_copy = np.prod(cf_set_copy)
                
    return cf_set_copy

def sum_product_eliminate_var(cf_set, variable):
    cf_set_var = []  # Ensemble des cf contenant la variable
    for c in cf_set:
        if variable in c:
            cf_set_var.append(c)
    cf_set = list(set(cf_set) - set(cf_set_var))
    cf_product = np.prod(cf_set_var)
    
    if(cf_set_var):
        cf_marg = cf_product.marginalize([variable])
        cf_set.append(cf_marg)

    return cf_set
