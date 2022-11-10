"""
activations.py

Contains bounds on various activation functions

Copyright (C) 2018, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Luca Daniel <dluca@mit.edu>
"""
from numba import njit
import numpy as np

#Functions for bounding various activation functions

@njit
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

@njit
def sigmoidd(x):
    return np.exp(-x)/(1.0+np.exp(-x))**2

@njit
def sigmoidid(x):
    return 2.0*np.arccosh(1.0/(2.0*np.sqrt(x)))



@njit 
def sigmoidut(l, u):
    act = sigmoid
    actd = sigmoidd
    actid = sigmoidid
    upper = u
    lower = 0
    al = act(l)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (act(guess)-al)/(guess-l)
        if guesss >= guesst:
            upper = guess
        else:
            lower = guess
    return upper
    
@njit 
def sigmoidlt(l, u):
    act = sigmoid
    actd = sigmoidd
    actid = sigmoidid
    upper = 0
    lower = l
    au = act(u)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (au-act(guess))/(u-guess)
        if guesss >= guesst:
            lower = guess
        else:
            upper = guess
    return lower


@njit 
def sigmoiduo(l, u):
    act = sigmoid
    actd = sigmoidd
    actid = sigmoidid
    upper = u
    lower = 0
    k = (act(l)-act(u))/(l-u)
    for i in range(20):
        guess = (upper + lower)/2
        #guessu = actd(lower)
        guessl = actd(guess)
        if guessl>k:
            lower = guess
        else:
            upper = guess
    return lower

@njit 
def sigmoidlo(l, u):
    act = sigmoid
    actd = sigmoidd
    actid = sigmoidid
    upper = 0
    lower = l
    k = (act(l)-act(u))/(l-u)
    for i in range(20):
        guess = (upper + lower)/2
        guessu = actd(guess)
        #guessl = actd(upper)
        if guessu>k:
            upper = guess
        else:
            lower = guess
    return lower

@njit
def tanh(x):
    return np.tanh(x)

@njit
def tanhd(x):
    return 1.0/np.cosh(x)**2

@njit
def tanhid(x):
    return np.arccosh(1.0/np.sqrt(x))

@njit 
def tanhut(l, u):
    act = tanh
    actd = tanhd
    actid = tanhid
    upper = u
    lower = 0
    al = act(l)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (act(guess)-al)/(guess-l)
        if guesss >= guesst:
            upper = guess
        else:
            lower = guess
    return upper
    
@njit 
def tanhlt(l, u):
    act = tanh
    actd = tanhd
    actid = tanhid
    upper = 0
    lower = l
    au = act(u)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (au-act(guess))/(u-guess)
        if guesss >= guesst:
            lower = guess
        else:
            upper = guess
    return lower



@njit 
def tanhuo(l, u):
    act = tanh
    actd = tanhd
    actid = tanhid
    upper = u
    lower = 0
    k = (act(l)-act(u))/(l-u)
    for i in range(20):
        guess = (upper + lower)/2
        #guessu = actd(lower)
        guessl = actd(guess)
        if guessl>k:
            lower = guess
        else:
            upper = guess
    return lower
    
@njit 
def tanhlo(l, u):
    act = tanh
    actd = tanhd
    actid = tanhid
    upper = 0
    lower = l
    k = (act(l)-act(u))/(l-u)
    for i in range(20):
        guess = (upper + lower)/2
        guessu = actd(guess)
        #guessl = actd(upper)
        if guessu>k:
            upper = guess
        else:
            lower = guess
    return lower

@njit
def atan(x):
    return np.arctan(x)

@njit
def atand(x):
    return 1.0/(1.0+x**2)

@njit
def atanid(x):
    return np.sqrt(1.0/x-1.0)

@njit 
def atanut(l, u):
    act = atan
    actd = atand
    actid = atanid
    upper = u
    lower = 0
    al = act(l)
    while abs(upper-lower)>0.0001:
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (act(guess)-al)/(guess-l)
        if guesss >= guesst:
            upper = guess
        else:
            lower = guess
    return upper
    
@njit 
def atanlt(l, u):
    act = atan
    actd = atand
    actid = atanid
    upper = 0
    lower = l
    au = act(u)
    while abs(upper-lower)>0.0001:
        guess = (upper + lower)/2
        guesst = actd(guess)
        if u-guess!=0:
            guesss = (au-act(guess))/(u-guess)
        if guesss >= guesst:
            lower = guess
        else:
            upper = guess
    return lower


@njit 
def atanuo(l, u):
    act = atan
    actd = atand
    actid = atanid
    upper = u
    lower = 0
    k = (act(l)-act(u))/(l-u)
    while abs(upper-lower)>0.0001:
        guess = (upper + lower)/2
        #guessu = actd(lower)
        guessl = actd(guess)
        if guessl>k:
            lower = guess
        else:
            upper = guess
    return lower
    
@njit 
def atanlo(l, u):
    act = atan
    actd = atand
    actid = atanid
    upper = 0
    lower = l
    k = (act(l)-act(u))/(l-u)
    while abs(upper-lower)>0.0001:
        guess = (upper + lower)/2
        guessu = actd(guess)
        #guessl = actd(upper)
        if guessu>k:
            upper = guess
        else:
            lower = guess
    return lower




@njit
def relu_linear_bounds(LB, UB):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)
    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                ## Original
                if LB[i,j,k] > 0:
                    alpha_u[i,j,k] = 1
                    alpha_l[i,j,k] = 1
                elif UB[i,j,k] <= 0:
                    pass #All zeros
                else:
                    alpha_u[i,j,k] = UB[i,j,k]/(UB[i,j,k]-LB[i,j,k])
                    alpha_l[i,j,k] = UB[i,j,k]/(UB[i,j,k]-LB[i,j,k])
                    beta_u[i,j,k] = -alpha_u[i,j,k]*LB[i,j,k]
    return alpha_u, alpha_l, beta_u, beta_l
@njit
def ada_linear_bounds(LB, UB):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)
    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                ## Adaptive
                if LB[i,j,k] >= 0:
                    alpha_u[i,j,k] = 1
                    alpha_l[i,j,k] = 1
                elif UB[i,j,k] <= 0:
                    pass #All zeros
                elif UB[i,j,k] >= -LB[i,j,k]:
                    alpha_u[i,j,k] = UB[i,j,k]/(UB[i,j,k]-LB[i,j,k])
                    alpha_l[i,j,k] = 1
                    beta_u[i,j,k] = -alpha_u[i,j,k]*LB[i,j,k]
                else:
                    alpha_u[i,j,k] = UB[i,j,k]/(UB[i,j,k]-LB[i,j,k])
                    beta_u[i,j,k] = -alpha_u[i,j,k]*LB[i,j,k]
    return alpha_u, alpha_l, beta_u, beta_l




@njit
def atan_linear_bounds(LB, UB):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)
    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                act = atan
                actd = atand
                actid = atanid
                actuo = atanuo
                actlo = atanlo
                actut = atanut
                actlt = atanlt
                ## General (Sigmoid-like functions)
                # 
                #print('---if start -----')
                if UB[i,j,k] < LB[i,j,k]: 
                    temp=LB[i,j,k]
                    LB[i,j,k]=UB[i,j,k]
                    UB[i,j,k]=temp
                elif UB[i,j,k]-LB[i,j,k]<0.001:
                    alpha_u[i,j,k] = actd(UB[i,j,k])
                    alpha_l[i,j,k] = actd(LB[i,j,k])
                    beta_u[i,j,k] = act(UB[i,j,k])-actd(UB[i,j,k])*UB[i,j,k]
                    beta_l[i,j,k] = act(LB[i,j,k])-actd(LB[i,j,k])*LB[i,j,k]
                else:
                    kk = (act(UB[i, j, k]) - act(LB[i, j, k])) / (UB[i, j, k] - LB[i, j, k])
                    ld=actd(LB[i, j, k])
                    ud=actd(UB[i, j, k])

                    if kk>ud:
                        alpha_u[i, j, k] = ud 
                        beta_u[i, j, k] = act(UB[i, j, k]) - alpha_u[i, j, k] * UB[i, j, k]
                    else:
                        alpha_u[i, j, k] = kk
                        beta_u[i, j, k] = act(UB[i, j, k]) - alpha_u[i, j, k] * UB[i, j, k]
                    if kk<ld:
                        alpha_l[i, j, k] = kk
                        beta_l[i, j, k] = act(LB[i,j,k]) - alpha_l[i, j, k] * LB[i,j,k]
                    
                    else:

                        alpha_l[i, j, k] = ld
                        beta_l[i, j, k] = act(LB[i,j,k]) - alpha_l[i, j, k] * LB[i,j,k]
    return alpha_u, alpha_l, beta_u, beta_l

@njit
def sigmoid_linear_bounds(LB, UB):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)
    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                act = sigmoid
                actd = sigmoidd
                actid = sigmoidid
                actlo = sigmoidlo
                actuo = sigmoiduo
                actut = sigmoidut
                actlt = sigmoidlt
                ## General (Sigmoid-like functions)
                # 
                if UB[i,j,k] < LB[i,j,k]: 
                    temp=LB[i,j,k]
                    LB[i,j,k]=UB[i,j,k]
                    UB[i,j,k]=temp
                elif UB[i,j,k]-LB[i,j,k]<0.001:
                    alpha_u[i,j,k] = actd(UB[i,j,k])
                    alpha_l[i,j,k] = actd(LB[i,j,k])
                    beta_u[i,j,k] = act(UB[i,j,k])-actd(UB[i,j,k])*UB[i,j,k]
                    beta_l[i,j,k] = act(LB[i,j,k])-actd(LB[i,j,k])*LB[i,j,k]
                else:
                    kk = (act(UB[i, j, k]) - act(LB[i, j, k])) / (UB[i, j, k] - LB[i, j, k])
                    ld=actd(LB[i, j, k])
                    ud=actd(UB[i, j, k])

                    if kk>ud:
                        alpha_u[i, j, k] = ud 
                        beta_u[i, j, k] = act(UB[i, j, k]) - alpha_u[i, j, k] * UB[i, j, k]
                    else:
                        alpha_u[i, j, k] = kk
                        beta_u[i, j, k] = act(UB[i, j, k]) - alpha_u[i, j, k] * UB[i, j, k]
                    if kk<ld:
                        alpha_l[i, j, k] = kk
                        beta_l[i, j, k] = act(LB[i,j,k]) - alpha_l[i, j, k] * LB[i,j,k]
                    
                    else:

                        alpha_l[i, j, k] = ld
                        beta_l[i, j, k] = act(LB[i,j,k]) - alpha_l[i, j, k] * LB[i,j,k]
    return alpha_u, alpha_l, beta_u, beta_l


@njit
def tanh_linear_bounds(LB, UB):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)
    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                act = tanh
                actd = tanhd
                actid = tanhid
                actuo = tanhuo
                actlo = tanhlo
                actut = tanhut
                actlt = tanhlt
                ## General (Sigmoid-like functions)
                if UB[i,j,k] < LB[i,j,k]: 
                    temp=LB[i,j,k]
                    LB[i,j,k]=UB[i,j,k]
                    UB[i,j,k]=temp
                elif UB[i,j,k]-LB[i,j,k]<0.001:
                    alpha_u[i,j,k] = actd(UB[i,j,k])
                    alpha_l[i,j,k] = actd(LB[i,j,k])
                    beta_u[i,j,k] = act(UB[i,j,k])-actd(UB[i,j,k])*UB[i,j,k]
                    beta_l[i,j,k] = act(LB[i,j,k])-actd(LB[i,j,k])*LB[i,j,k]
                else:
                    kk = (act(UB[i, j, k]) - act(LB[i, j, k])) / (UB[i, j, k] - LB[i, j, k])
                    ld=actd(LB[i, j, k])
                    ud=actd(UB[i, j, k])

                    if kk>ud:
                        alpha_u[i, j, k] = ud 
                        beta_u[i, j, k] = act(UB[i, j, k]) - alpha_u[i, j, k] * UB[i, j, k]
                    else:
                        alpha_u[i, j, k] = kk
                        beta_u[i, j, k] = act(UB[i, j, k]) - alpha_u[i, j, k] * UB[i, j, k]
                    if kk<ld:
                        alpha_l[i, j, k] = kk
                        beta_l[i, j, k] = act(LB[i,j,k]) - alpha_l[i, j, k] * LB[i,j,k]
                    
                    else:

                        alpha_l[i, j, k] = ld
                        beta_l[i, j, k] = act(LB[i,j,k]) - alpha_l[i, j, k] * LB[i,j,k]
    return alpha_u, alpha_l, beta_u, beta_l
