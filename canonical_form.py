# coding: utf-8

import numpy as np
import continuous_variable as cv
import variable_elimination as ve
from numpy.linalg import inv
from numpy.linalg import det


class Scope:

    def __init__(self, variables=[]):
        self.__v = variables
        self.__size = len(variables)
        self.__variableIds = {variables[i] : i for i in range(self.__size)}
    
    def __str__(self):
        return str(self.__v)
    
    def __contains__(self, item):
        return item in self.__v

    def get_v(self):
        return self.__v
    
    def get_variableIds(self):
        return self.__variableIds


class CanonicalForm:
    
    # Il faudra verifier pour l'initialisation par defaut
    def __init__(self, scope=[], K=[[0]], h=[0], g=0):
        # On s'attend à ce que le scope soit passé comme une liste de variables
        # continues.
        # Il faudrait regarder que la cf identité fonctionne correctement
        self.__scope = Scope(scope)
        self.__K = K
        self.__h = h
        self.__g = g

    def __str__(self):
        return "scope = {0}, K = {1}, h = {2}, g = {3}".format(self.__scope,
                                                               self.__K,
                                                               self.__h,
                                                               self.__g)
    
    def __repr__(self):
        return str(self)

    def __contains__(self, item):
        return item in self.__scope

    def get_scope(self):
        return self.__scope
    
    @classmethod
    def from_gaussian(cls, scope, mu, Sigma):
        K = inv(Sigma)
        h = np.dot(K, mu)
        g = - 0.5 * np.dot( np.dot(np.transpose(mu), K), mu ) \
            - np.log( (2*np.pi)**(0.5*len(scope)) * det(Sigma)**0.5)
        return cls(scope, K, h, g)
    
    @classmethod
    def from_cond_gaussian(cls, uncond_scope, cond_scope, mu, Sigma, B):
        scope = uncond_scope + cond_scope
        scope.sort()
        
        M = []
        i = 0
        j = 0
        for v in scope:
            if v in uncond_scope:
                M.append(np.eye(1, len(Sigma), i))
                i += 1
            elif v in cond_scope:
                M.append(-B[j])
                j += 1
                
        M = np.array(M)
        
        invSigma = inv(Sigma)
        K = np.dot(np.dot(M, invSigma), np.transpose(M))
        h = np.dot(M, np.dot(invSigma, mu))
        g = - 0.5 * np.dot( np.dot(np.transpose(mu), invSigma), mu ) \
            - np.log( (2*np.pi)**(0.5*len(Sigma)) * det(Sigma)**0.5)
        return cls(scope, K, h, g)

    def to_gaussian(self):
        # La conversion en dtype float est necessaire sinon
        # une erreur tres mysterieuse apparait...
        # Elle est apparue d'un coup alors que tout fonctionnait !?
        # Ca viendrait du fait que les array soient de type object
        # il faudrait donc chercher a quelle etape il y a cette conversion
        self.__K = np.array(self.__K, dtype=float)
        self.__h = np.array(self.__h, dtype=float)
        self.__g = np.array(self.__g, dtype=float)
        Sigma = inv(self.__K)
        mu = np.dot(Sigma, self.__h)
        c = np.exp(self.__g +
                   0.5 * np.dot( np.dot(np.transpose(self.__h), Sigma), self.__h)
                  )
        return Sigma, mu, c


    def __mul__(self, other):
        # Ce serait plus intelligent de mettre l'opération d'extension
        # dans une fonction à part.
        # Il faudrait également prendre en compte le cas ou le scope est vide
        # car sinon il y a une erreur
        scope = list(set(self.__scope.get_v() + other.__scope.get_v()))
        scope.sort()
        v_add_self = list(set(scope) - set(self.__scope.get_v()))
        v_add_self.sort()
        v_add_other = list(set(scope) - set(other.__scope.get_v()))
        v_add_other.sort()
        
        
        if not self.__K.size and not self.__h.size:
            augmented_K_self = np.zeros((len(v_add_self), len(v_add_self)))
            augmented_h_self = np.zeros((len(v_add_self), 1))
        else:
            augmented_K_self = self.__K
            augmented_h_self = self.__h
            for e in v_add_self:
                pos = scope.index(e)
                if pos < len(augmented_K_self):
                    augmented_K_self = np.insert(augmented_K_self,
                                                 [pos],
                                                 np.zeros((1,len(augmented_K_self))),
                                                 axis=0)
                
                    augmented_K_self = np.insert(augmented_K_self,
                                                 [pos],
                                                 np.zeros((len(augmented_K_self),1)),
                                                 axis=1)
                    augmented_h_self = np.insert(augmented_h_self, pos, 0, axis=0)
                
                else:
                    augmented_K_self = np.insert(augmented_K_self,
                                                 [len(augmented_K_self[0])],
                                                 np.zeros((1,len(augmented_K_self))),
                                                 axis=0)
                    augmented_K_self = np.insert(augmented_K_self,
                                                 [len(augmented_K_self[1])],
                                                 np.zeros((len(augmented_K_self),1)),
                                                 axis=1)
                    augmented_h_self = np.append(augmented_h_self, [[0]], axis=0)
        
        if not other.__K.size and not other.__h.size:
            augmented_K_other = np.zeros((len(v_add_other), len(v_add_other)))
            augmented_h_other = np.zeros((len(v_add_other), 1))
        else:
            augmented_K_other = other.__K
            augmented_h_other = other.__h
            for e in v_add_other:
                pos = scope.index(e)
                if pos < len(augmented_K_other):
                    augmented_K_other = np.insert(augmented_K_other,
                                                  [pos],
                                                  np.zeros((1,len(augmented_K_other))),
                                                  axis=0)
                
                    augmented_K_other = np.insert(augmented_K_other,
                                                  [pos],
                                                  np.zeros((len(augmented_K_other),1)),
                                                  axis=1)
                    augmented_h_other = np.insert(augmented_h_other, pos, 0, axis=0)
                    
                else:
                    augmented_K_other = np.insert(augmented_K_other,
                                                  [len(augmented_K_other[0])],
                                                  np.zeros((1,len(augmented_K_other))),
                                                  axis=0)
                
                    augmented_K_other = np.insert(augmented_K_other,
                                                  [len(augmented_K_other[1])],
                                                  np.zeros((len(augmented_K_other),1)),
                                                  axis=1)
                    augmented_h_other = np.append(augmented_h_other, [[0]], axis=0)
        
        K = augmented_K_self + augmented_K_other
        h = augmented_h_self + augmented_h_other
        g = self.__g + other.__g

        return CanonicalForm(scope, K, h, g)


    def marginalize(self, scope):
        
        if scope is []:
            return self

        # Pour eviter l'erreur
        self.__K = np.array(self.__K, dtype=float)
        self.__h = np.array(self.__h, dtype=float)
        self.__g = np.array(self.__g, dtype=float)
        
        sum_idx = [self.__scope.get_variableIds()[v] for v in scope]
        unsum_idx =   set(self.__scope.get_variableIds().values()) \
                    - set(sum_idx)
        unsum_idx = list(unsum_idx)
        sum_idx.sort()
        unsum_idx.sort()

        K_xx = self.__K[np.ix_(unsum_idx,unsum_idx)]
        K_yy = self.__K[np.ix_(sum_idx,sum_idx)]
        K_xy = self.__K[np.ix_(unsum_idx,sum_idx)]
        K_yx = self.__K[np.ix_(sum_idx,unsum_idx)]        

        h_x = self.__h[unsum_idx]
        h_y = self.__h[sum_idx]

        K = K_xx - np.dot(np.dot(K_xy, inv(K_yy)), K_yx)
        h = h_x - np.dot(np.dot(K_xy, inv(K_yy)), h_y)
        g = self.__g + 0.5*(len(scope)*np.log(2*np.pi) - np.log(det(K_yy)) + \
                            np.dot(np.dot(np.transpose(h_y), inv(K_yy)), h_y))

        scope = list(set(self.__scope.get_v()) - set(scope))
        scope.sort()
        
        return CanonicalForm(scope, K, h, g)


    def reduce(self, scope, values):

        red_idx = [self.__scope.get_variableIds()[v] for v in scope]
        unred_idx =  set(self.__scope.get_variableIds().values()) \
                         - set(red_idx)
        unred_idx = list(unred_idx)
        unred_idx.sort()

        K_xx = self.__K[np.ix_(unred_idx,unred_idx)]
        K_yy = self.__K[np.ix_(red_idx,red_idx)]
        K_xy = self.__K[np.ix_(unred_idx,red_idx)]
        
        h_x = self.__h[unred_idx]
        h_y = self.__h[red_idx]

        h = h_x - np.dot(K_xy, values)

        g = self.__g + np.dot(np.transpose(h_y), values) \
                     - 0.5 * np.dot(np.dot(np.transpose(values), K_yy), values)

        scope = list(set(self.__scope.get_v()) - set(scope))
        scope.sort()

        return CanonicalForm(scope, K_xx, h, g)
#
#    def __truediv__(self, other):
#        scope = self.__scope + other.__scope
#        K = self.__K - other.__K
#        h = self.__h - other.__h
#        g = self.__g - other.__g
#
#        return CanonicalForm(scope, K, h, g)

if __name__ == '__main__':
    X = cv.ContinuousVariable('X')
    Y = cv.ContinuousVariable('Y')
    Z = cv.ContinuousVariable('Z')
    T = cv.ContinuousVariable('T')
    
    
    h1 = np.array([[1], [-1]])
    h2 = np.array([[5], [-1]])
    
    g1 = -3
    g2 = 1
    
    K1 = np.array([[1, -1], [-1, 1]])
    K2 = np.array([[3, -2], [-2, 4]])
    
    scope1 = [X, Y]
    scope2 = [Y, Z]
    
    cf1 = CanonicalForm(scope1, K1, h1, g1)
    cf2 = CanonicalForm(scope2, K2, h2, g2)
    
    cf = cf1 * cf2
    cfr1m = cf.marginalize([Y])
    print('Marginale sur Y', cfr1m)
    cfr1 = cfr1m.reduce([Z], [[2]])
    print('Reduction sur Z', cfr1)
    cf1s = cf1 * cf1

    mu = np.array([1, 1])
    Sigma = np.array([[1, 0.5], [0.5, 1]])
    cf4 = CanonicalForm.from_gaussian(scope1, mu, Sigma)
    print('cf4', cf4)
    cf4 = cf4.marginalize([X])
    print(cf4)
    m, S, c = cf4.to_gaussian()
    print(m, S, c)

    mu = np.array([0])
    Sigma = np.array([[0.002]])
    B = np.array([[1],[1]])
    print(mu, Sigma, B);
    cflg = CanonicalForm.from_cond_gaussian([Y], [X,Z], mu, Sigma, B)
    print(cflg)
    
    cflg = CanonicalForm.from_cond_gaussian([Z],[Y], np.array([[50]]),
                                                     np.array([[10]]),
                                                     np.array([[-100]]))
    cfy = CanonicalForm.from_gaussian([Y], np.array([[20]]), np.array([[9]]))

    print("cflg : ", cflg)
    print("cfy : ", cfy)
    cf_red = cflg.reduce([Z], [[18.]])
    cf_prod = cf_red * cfy

    print("elim : ", cf_prod)

    # cf_from_lg = CanonicalForm.from_cond_gaussian([Y], [Y,Z], 
    
    
#    scope1 = [X, T]
#    scope2 = [Y, Z]
#    
#    cf1 = CanonicalForm(scope1, K1, h1, g1)
#    cf2 = CanonicalForm(scope2, K2, h2, g2)
#    
#    cf = cf1 * cf2
#    cf1s = cf1 * cf1
#    
#    cfm = cf.marginalize([X, Y])
#    
#    cfr = cf.reduce([X,Z], [[0.1], [0.2]])
#    
#    mu = np.array([[3]])
#    Sigma = np.array([[1]])
#    B = np.array([[1], [1]])
#    cfc = CanonicalForm.from_cond_gaussian([X, Y, Z], mu, Sigma, B)
