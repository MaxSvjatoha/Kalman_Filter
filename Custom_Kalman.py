# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:38:06 2021

@author: m.svjatoha
"""

from Kalman import gauss_pdf, kf_predict as predict, kf_update as update
from sys import exit

import numpy as np

# Position estimation based on x = x_o + (v_o * dt) + (a * 1/2 t ^ 2)

# Given the above and time variables...:
t_0 = 0
t = t_0
dt = 1e-1

# ... gives the following transition matrix:

A = np.array([[1, t, 0.5 * (t**2)],
              [0, 1,            t],
              [0, 0,            1]])

# Initial states, assuming X = [acceleration, velocity and position]

X = np.array([[0.0], [0.0], [0.0]])

# = Initial state covariance matrix:
    
P = np.diag((0.01, 0.01, 0.01))
print(P)

Q = np.eye(X.shape[0])          # Process noise covariable matrix.
B = np.eye(X.shape[0])          # Input effect matrix
U = np.zeros((X.shape[0],1))    # The control input

print(Q)
print(B)
print(U)