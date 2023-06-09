#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:06:34 2021
This is the script for MLRSSC under all views
@author: Jiani Ma
"""
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
    invmat = invmat.astype(np.float)
    fore_mat = LA.inv(invmat)
    behind_mat = mu1 * K + (mu2) * (C2_mat-lambda2_mat/mu2) + (mu3) * (C1_mat-lambda3_mat/mu3) + (mu4) * (C3_mat-lambda4_mat/mu4) + np.dot(X_mat.T,lambda1_mat)
    A_mat = np.dot(fore_mat, behind_mat)
    return A_mat



def parse_args():    
    parser = argparse.ArgumentParser(description="This is a script for MLRSSC.")
    parser.add_argument('-mu', '--mu_topofallfeature', type=float, nargs='?', default=10,help='Performing MLRSSC.')
    return parser.parse_args()  


if __name__=="__main__":
    args=parse_args()
    mu = args.mu_topofallfeature
    beta1 = 0.5
    lambdav = 0.3
    beta2 = 0.5
    mu_max = 1e6
    pho = 1.5
    iter_max = 100
    err_th = 1e-6
    
    # Drug Features or Target features
  
    fou = pd.read_excel("/home/jiani.ma/DTI/MVGCN_Optimal/dataset/protein_information/protein_vector_d400.xlsx").values   # 708 * 100
    kar = pd.read_excel("/home/jiani.ma/DTI/MVGCN_Optimal/dataset/protein_information/mat_protein_disease.xlsx").values   # 708 * 5603
    #fac = pd.read_excel("D:/DTI/DTINet_data/drug_information/mat_drug_side_effect.xlsx").values  # 708 * 4192

    X1 = fou.T  #(76,2000)
    X2 = kar.T   #(64,2000)
    #X3 = kar.T   #(216,2000)
    X = []
    X.append(X1)
    X.append(X2)
    #X.append(X3)
    n = X1.shape[1]  # samples number
    nv = len(X)      #view number
        
    lambda1 = []
    lambda11_mat = np.zeros((X1.shape[0],X1.shape[1]))
    lambda12_mat = np.zeros((X2.shape[0],X2.shape[1]))
    #lambda13_mat = np.zeros((X3.shape[0],X3.shape[1]))
    lambda1.append(lambda11_mat)
    lambda1.append(lambda12_mat)
    #lambda1.append(lambda13_mat)

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
        #print("iter",iter)
        #print("mu",mu)
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
            #print("err_AC1",err1)
            #print("err_AC2",err2)
            #print("err_AC3",err3)
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
    
    # drug_af = pd.DataFrame(af)
    # writer = pd.ExcelWriter("D:/DTI/DTINet_data/target_information/target_affinity_mat.xlsx")
    # drug_af.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
    # writer.save()
    
