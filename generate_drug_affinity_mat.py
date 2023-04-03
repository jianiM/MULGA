#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt  
import argparse
from numpy.linalg import norm
from random import normalvariate
from config_init import get_config


def svt(A,tol):
    U,s,VT = LA.svd(A,full_matrices=False)
    shrink_s = s - tol
    shrink_s[shrink_s<0]=0
    S = np.diag(shrink_s)
    svtm_tmp = np.dot(U,S)
    svtm = np.dot(svtm_tmp,VT)
    return svtm
    
def C1_mat_update(A_mat,lambda3_mat, mu3, beta1):
    tol = beta1 / mu3
    lambda_mu_3 = lambda3_mat / mu3
    C1_tmp_mat = A_mat + lambda_mu_3
    C1_mat = svt(C1_tmp_mat,tol)
    return C1_mat

def C2_mat_update(A_mat,lambda2_mat,mu2,beta2):
    thresh = beta2 / mu2 
    lambda_mu_2 = lambda2_mat / mu2
    C2_tmp_mat = A_mat + lambda_mu_2
    Y1 = C2_tmp_mat-thresh
    Y1[Y1<0] = 0
    Y2 = C2_tmp_mat+thresh
    Y2[Y2>0] = 0 
    C2_mat = Y1 + Y2
    return C2_mat

def C3_mat_update(lambdav,nv,mu4, A_mat,lambda4_mat,C_sum_mat):
    invcoeff = 2 * lambdav * (nv-1) + mu4
    coeff = 1.0 / (invcoeff+0.0001)
    C3_tmp_mat = 2 * lambdav * C_sum_mat + mu4 * A_mat + lambda4_mat 
    C3_mat =  coeff * C3_tmp_mat
    return C3_mat

def A_mat_update(X_mat, mu1, mu2, mu3, mu4, C1_mat, C2_mat, C3_mat, lambda1_mat, lambda2_mat, lambda3_mat, lambda4_mat):
    K = np.dot(X_mat.T,X_mat)
    invmat = (mu1) * K + (mu2 + mu3 + mu4)*np.identity(C1_mat.shape[0])
    invmat = invmat.astype(np.float64)
    fore_mat = LA.inv(invmat)
    behind_mat = mu1 * K + (mu2) * (C2_mat-lambda2_mat/mu2) + (mu3) * (C1_mat-lambda3_mat/mu3) + (mu4) * (C3_mat-lambda4_mat/mu4) + np.dot(X_mat.T,lambda1_mat)
    A_mat = np.dot(fore_mat, behind_mat)
    return A_mat



if __name__=="__main__": 
    config = get_config()
    root_path = config.root_path_topofallfeature
    dataset = config.dataset_topofallfeature
    macc_file_path = config.drug_macc_file_topofallfeature
    mogan_file_path = config.drug_mogan_file_topofallfeature
    rdk_file_path = config.drug_rdk_file_topofallfeature
    
    mu = 10
    beta1 = 0.7     # determining rank 
    lambdav = 0.3
    beta2 = 0.3     # determining sparsity
    mu_max = 1e6
    pho = 1.2
    iter_max = 100
    err_th = 1e-6
        
    # Drug Features or Target features 
    macc_path = os.path.join(root_path,dataset,macc_file_path)
    mogan_path = os.path.join(root_path,dataset,mogan_file_path)
    rdk_path = os.path.join(root_path,dataset,rdk_file_path)
    
    macc = pd.read_excel(macc_path,header=None).values    
    mogan = pd.read_excel(mogan_path,header=None).values   
    rdk = pd.read_excel(rdk_path,header=None).values  

   
    X1 = macc.T  
    X2 = mogan.T   #(64,2000)
    X3 = rdk.T   #(216,2000)

    X = []
    X.append(X1)
    X.append(X2)
    X.append(X3)
    n = X1.shape[1]  # samples number
    nv = len(X)      #view number
    
    lambda1 = []
    lambda11_mat = np.zeros((X1.shape[0],X1.shape[1]))
    lambda12_mat = np.zeros((X2.shape[0],X2.shape[1]))
    lambda13_mat = np.zeros((X3.shape[0],X3.shape[1]))

    lambda1.append(lambda11_mat)
    lambda1.append(lambda12_mat)
    lambda1.append(lambda13_mat)

    """
    don't need to revise 
    """

    A = np.zeros((nv,n,n))
    C1 = np.zeros((nv,n,n))
    C2 = np.zeros((nv,n,n))
    C3 = np.zeros((nv,n,n))
    lambda2 = np.zeros((nv,n,n))
    lambda3 = np.zeros((nv,n,n))
    lambda4 = np.zeros((nv,n,n))
    A_prev = np.zeros((nv,n,n))
    iter = 0 
    converged = False
    
    while (iter<100) and (not converged):
        iter = iter + 1
        print("iter",iter)
        print("mu",mu)
        mu1 = mu    
        mu2 = mu
        mu3 = mu 
        mu4 = mu        
        C_sum = np.zeros((nv,n,n))
               
        for v in range(nv): 
            for v_tmp in range(nv):
                if v_tmp != v: 
                    C_sum[v] = C_sum[v] + C2[v_tmp]
        
        for i in range(nv): 
            A_prev[i] = A[i]
            A[i] = A_mat_update(X[i], mu1, mu2, mu3, mu4, C1[i], C2[i], C3[i], lambda1[i], lambda2[i], lambda3[i], lambda4[i])
            C2[i] = C2_mat_update(A[i],lambda2[i],mu2,beta2)
            C2[i] = C2[i] - np.diag(np.diagonal(C2[i]))            
            C1[i] = C1_mat_update(A[i],lambda3[i], mu3, beta1)            
            C3[i] = C3_mat_update(lambdav,nv,mu4, A[i],lambda4[i],C_sum[i])
            lambda1[i] = lambda1[i] + mu1 * (X[i]-np.dot(X[i],A[i]))
            lambda2[i] = lambda2[i] + mu2 * (A[i]-C2[i])
            lambda3[i] = lambda3[i] + mu3 * (A[i]-C1[i])
            lambda4[i] = lambda4[i] + mu4 * (A[i]-C3[i])               
        
      
        converged = True
        
        for j in range(nv):
            err1 = np.max(abs(A[j]-C1[j]))
            err2 = np.max(abs(A[j]-C2[j]))
            err3 = np.max(abs(A[j]-C3[j]))
            err4 = np.max(abs(A_prev[j]-A[j]))

            print("err_AA",err4)
            if (err1>err_th) or (err2>err_th) or (err3>err_th) or (err4>err_th): 
                converged = False
                break
        mu = min(pho*mu,mu_max)
        
    C_avg = np.zeros((n,n))
    for v in range(nv):
        C_avg = C_avg + C2[v]
        
    C_avg = C_avg / nv        
    af = abs(C_avg) + abs(C_avg.T)
    for i in range(af.shape[0]):
        for j in range(af.shape[1]):
            print(af[i][j])
    
