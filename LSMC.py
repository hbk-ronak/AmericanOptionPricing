import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
PATHS = 100000
k = 4

def SimulateGBM(S0, r, sd, T, paths, steps, reduce_variance = True):
    steps = int(steps)
    dt = T/steps
    Z = np.random.normal(0, 1, paths//2 * steps).reshape((paths//2, steps))
    # Z_inv = np.random.normal(0, 1, paths//2 * steps).reshape((paths//2, steps))
    if reduce_variance:
      Z_inv = -Z
    else:
      Z_inv = np.random.normal(0, 1, paths//2 * steps).reshape((paths//2, steps))
    dWt = math.sqrt(dt) * Z
    dWt_inv = math.sqrt(dt) * Z_inv
    dWt = np.concatenate((dWt, dWt_inv), axis=0)
    St = np.zeros((paths, steps + 1))
    St[:, 0] = S0
    for i in range (1, steps + 1):
        St[:, i] = St[:, i - 1]*np.exp((r - 1/2*np.power(sd, 2))*dt + sd*dWt[:, i - 1])
    
    return St[:, 1:]

def power_polynomials(S, k):
    
    #  the first k terms of Laguerre Polynomials (k<=4)
    x1 = np.ones(S.shape)
    x2 = S
    x3 = S**2
    x4 = S**3 
    
    X  = [np.stack([x1, x2], axis = 1),
          np.stack([x1, x2, x3], axis = 1),
          np.stack([x1, x2, x3, x4], axis = 1)]
    
    return X[k-2]

def laguerre_polynomials(S, k):
    
    #  the first k terms of Laguerre Polynomials (k<=4)
#    x1 = np.exp(-S/2)
#    x2 = np.exp(-S/2) * (1 - S)
#    x3 = np.exp(-S/2) * (1 - 2*S + S**2/2)
#    x4 = np.exp(-S/2) * (1 - 3*S + 3* S**2/2 - S**3/6)

    u0 = np.ones(S.shape)
    x1 = 1 - S
    x2 = 1 - 2*S + S**2/2
    x3 = 1 - 3*S + 3*S**2/2 - S**3/6
    x4 = 1 - 4*S + 3*S**2 - 2*S**3/3 + S**4/24

    X  = [np.stack([u0, x1, x2], axis = 1),
          np.stack([u0, x1, x2, x3], axis = 1),
          np.stack([u0, x1, x2, x3, x4], axis = 1)]
    
    return X[k-2]

def hermite_polynomials(S, k):
    
    #  the first k terms of Laguerre Polynomials (k<=4)
    x1 = np.ones(S.shape)
    x2 = 2*S
    x3 = 4*S**2 - 2
    x4 = 8*S**3 - 12
    
    X  = [np.stack([x1, x2], axis = 1),
          np.stack([x1, x2, x3], axis = 1),
          np.stack([x1, x2, x3, x4], axis = 1)]
    
    return X[k-2]

def chebychev1_polynomials(S, k):
    u0 = np.ones(S.shape)
    x1 = S
    x2 = 2*S**2 - 1
    x3 = 4*S**3 - 3*S
    x4 = 8*S**4 - 8*S**2 + 1
    x5 = 16*S**5 - 20*S**3 + 5*S
        
    X  = [np.stack([u0, x1, x2], axis = 1),
          np.stack([u0, x1, x2, x3], axis = 1),
          np.stack([u0, x1, x2, x3, x4], axis = 1)]
    
    return X[k-2]   

def chebychev2_polynomials(S, k):
    u0 = np.ones(S.shape)
    x1 = 2*S 
    x2 = 4*S**2 - 1
    x3 = 8*S**3-4*S
    x4 = 16*S**4 - 12*S**2 + 1
    x5 = 32*S**5 - 32*S**3 + 6*S        
    
    X  = [np.stack([u0, x1, x2], axis = 1),
          np.stack([u0, x1, x2, x3], axis = 1),
          np.stack([u0, x1, x2, x3, x4], axis = 1)]
    
    return X[k-2] 

def legendre_polynomials(S, k):
    u0 = np.ones(S.shape)
    x1 = S 
    x2 = (3*S**2 - 1)/2
    x3 = (5*S**3 - 3*S)/2
    x4 = (35*S**4 - 30*S**2 + 3)/8
    x5 = (63*S**5 - 70*S**3 + 15*S)/8
    
    X  = [np.stack([u0, x1, x2], axis = 1),
          np.stack([u0, x1, x2, x3], axis = 1),
          np.stack([u0, x1, x2, x3, x4], axis = 1)]
    
    return X[k-2]     

def priceOption(S0, K, r, paths, sd, T, steps, Stock_Matrix, reduce_variance = True):
  steps = int(steps)
  Stn = Stock_Matrix/K
  dt = T/steps
  cashFlow = np.zeros((paths, steps))
  cashFlow[:,steps - 1] = np.maximum(1 - Stn[:,steps - 1], 0)
      
  cont_value = cashFlow

  decision = np.zeros((paths, steps))
  decision[:, steps - 1] = 1

  discountFactor = np.tile(np.exp(-r*dt* np.arange(1, 
                                      steps + 1, 1)), paths).reshape((paths, steps))
  for i in reversed(range(steps - 1)):

          # Find in the money paths
          in_the_money_n = np.where(1 - Stn[:, i] > 0)[0]
          out_of_money_n = np.asarray(list(set(np.arange(paths)) - set(in_the_money_n)))
          

          X = legendre_polynomials(Stn[in_the_money_n, i], k)
          Y = cashFlow[in_the_money_n, i + 1]/np.exp(r*dt)

          A = np.dot(X.T, X)
          b = np.dot(X.T, Y)
          Beta = np.dot(np.linalg.pinv(A), b)

          cont_value[in_the_money_n,i] =  np.dot(X, Beta)
          try:
              cont_value[out_of_money_n,i] =  cont_value[out_of_money_n, i + 1]/np.exp(r*dt)
          except:
              pass

          decision[:, i] = np.where(np.maximum(1 - Stn[:, i], 0)  - cont_value[:,i] >= 0, 1, 0)
          cashFlow[:, i] =  np.maximum(1 - Stn[:, i], cont_value[:,i])
                  


  first_exercise = np.argmax(decision, axis = 1) 
  decision = np.zeros((len(first_exercise), steps))
  decision[np.arange(len(first_exercise)), first_exercise] = 1
  last = np.sum(decision*discountFactor*cashFlow*K, axis = 1)
  option_value = np.mean(last)
  var = np.sum((last-option_value)**2)/(last.shape[0]-1)
  return option_value,var


Stock_Matrix = SimulateGBM(40, 0.06, 0.5, 1, PATHS, 250, True)
price_reduced, var_reduced = priceOption(40,i,0.06,PATHS,.5, 1, 250, Stock_Matrix, True)