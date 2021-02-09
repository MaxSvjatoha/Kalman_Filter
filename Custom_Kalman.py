# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:38:06 2021

@author: m.svjatoha
"""

from Kalman import gauss_pdf, kf_predict as predict, kf_update as update
from sys import exit

import numpy as np

def normal_noise(sigma, mu):
    noise = sigma * np.random.randn(1) + mu
    return noise[0]


# Position estimation based on x = x_o + (v_o * dt) + (a * 1/2 t ^ 2)

# Given the above and time variables...:
t_0 = 0
t = t_0
dt = 1e-1

# ... gives the following transition matrix:

A = np.array([[1, t, 0.5 * (t**2)],
              [0, 1,            t],
              [0, 0,            1]])

# Alternative, a stationary system:

# A = np.array([[1, 0, 0],
#               [0, 1, 0],
#               [0, 0, 1]])

# Initial states, assuming X = [position, velocity]

X = np.array([[0.0], [0.0], [0.0]])

# = Initiate state covariance matrix:
    
P = np.diag((0.01, 0.01, 0.01))

Q = np.eye(X.shape[0])          # Process noise covariable matrix.
B = np.eye(X.shape[0])          # Input effect matrix
U = np.zeros((X.shape[0],1))    # The control input

# Measurement matrices:

Y = np.array([[X[0,0] + normal_noise(1, 0)], [X[1,0] +\
    normal_noise(1, 0)], [X[2,0] +\
    normal_noise(1, 0)]]) # Measured output.
C = np.array([[1, 0, 0]]) # The C from C*Y in the state space representation.
R = np.eye(Y.shape[0]) # The R from C*Y+R.

# print(Y)
# print(R)
# print(np.dot(C, np.dot(P, C.T)))

# Applying the Kalman Filter

N_iter = 50

for i in np.arange(0, N_iter):
    (X, P) = predict(X, P, A, Q, B, U)
    (X, P, K, IM, IS, LH) = update(X, P, Y, C, R)
    Y = np.array([[X[0,0] + normal_noise(1, 0)], [X[1,0] +\
        normal_noise(1, 0)]])
    print(X)
    print(Y)