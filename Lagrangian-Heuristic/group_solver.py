from errno import ENETUNREACH
import numpy as np
import torch
import torch.nn as nn
from pruners.utils import *
import numpy.linalg as la
import numba as nb
from time import time
from sklearn.utils import extmath
from collections import namedtuple
import warnings
import copy
import time
from numba.core.errors import NumbaDeprecationWarning, \
    NumbaPendingDeprecationWarning, NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
from numba import prange
from torch.utils.data import DataLoader
import L0_card
import os


def max_submatrix_sum(A, r, c, max_iter=100):
    # Find a submatrix of A with r rows and c columns such that the sum of elements in submatrix is maximized
    # Alternating maximization method (heurisitc)
    
    m, n = A.shape
    rows = np.random.choice(np.arange(m), size = r, replace=False)
    cols = np.argsort(np.sum(A[rows,:], axis=0))[n-c:n]
    for i in range(max_iter):
        rows_prev, cols_prev = rows, cols
        rows = np.argsort(np.sum(A[:,cols], axis=1))[m-r:m]
        cols = np.argsort(np.sum(A[rows,:], axis=0))[n-c:n]
        if set(rows)==set(rows_prev) and set(cols)==set(cols_prev):
            break
            
    return np.sort(rows), np.sort(cols)


def proj_group(w, size_list, prune_list):
    
    i_w = 0
    for i_s in range(len(size_list)):
        
        param_size = size_list[i_s]
        prune_size = prune_list[i_s]
        count = np.prod(param_size)
        w_cur = np.copy(w[i_w:i_w+count]).reshape(param_size)
        if len(w_cur.shape) == 4:
            w_cursum = np.sum(np.sum(np.abs(w_cur),axis=3),axis=2)
            rows, cols = max_submatrix_sum(w_cursum, prune_size[0], prune_size[1], 100)
            rows = np.array(list(set(np.arange(param_size[0])) - set(rows)))
            cols = np.array(list(set(np.arange(param_size[1])) - set(cols)))
            if len(rows) > 0:
                w_cur[rows,:,:,:] = 0
            if len(cols) > 0:
                w_cur[:,cols,:,:] = 0
        else: 
            w_cursum = np.abs(w_cur)
            rows, cols = max_submatrix_sum(w_cursum, prune_size[0], prune_size[1], 100)
            rows = np.array(list(set(np.arange(param_size[0])) - set(rows)))
            cols = np.array(list(set(np.arange(param_size[1])) - set(cols)))
            if len(rows) > 0:
                w_cur[rows,:] = 0
            if len(cols) > 0:
                w_cur[:,cols] = 0
        
        w[i_w:i_w+count] = w_cur.reshape(-1)
        i_w = i_w + count
        
    return w



def deter_groupsupp(w1, k, model, modules_to_prune, prune_type):
    
    sparsity = 1-k/len(w1)
    # get size
    size_list = []   
    state_dict = model.state_dict()
    for para in modules_to_prune:
        size_list.append(state_dict[para].shape)
        
    prune_list = []
    if prune_type == "overlap":
        prune_ratio = np.sqrt(1-sparsity)
    elif prune_type == "channel" or prune_type == "filter":
        prune_ratio = 1-sparsity
        
        
    total_para = 0
    for si in range(len(size_list)):
        param_size = size_list[si]
        
        if len(param_size) == 2:
            total_para += np.prod(param_size) 
            prune_list.append([param_size[0], param_size[1]])
            continue
            
        outsize, insize = param_size[0], param_size[1]    
        if (prune_type == "overlap" or prune_type == "channel") and insize >= 4:
            insize = int(np.floor(insize * prune_ratio))
        if (prune_type == "overlap" or prune_type == "filter"):
            outsize = int(np.floor(outsize * prune_ratio))
            
        prune_list.append([outsize, insize])
        total_para += np.prod(param_size[2:]) * (insize) * (outsize) 

    for si in range(len(size_list)):
        param_size = size_list[si]
        
        if len(param_size) == 2:
            continue
            
        outsize, insize = param_size[0], param_size[1]    
        if (prune_type == "overlap" or prune_type == "channel"):
            
            if prune_list[si][1] < param_size[1] and total_para + np.prod(param_size[2:]) * (prune_list[si][0])  <= k:
                prune_list[si][1] += 1
                total_para += np.prod(param_size[2:]) * (prune_list[si][0])
                
        if prune_type == "overlap" or prune_type == "filter":
            
            if prune_list[si][0] < param_size[0] and total_para + np.prod(param_size[2:]) * (prune_list[si][1]) <= k:
                prune_list[si][0] += 1
                total_para += np.prod(param_size[2:]) * (prune_list[si][1])
                
    # generate support
    beta = proj_group(np.copy(w1), size_list, prune_list)
                
    return beta
    
    

def group_pruner(y,X,w1,k,alpha,lambda2, beta_tilde2, model, modules_to_prune, block_diag, solve_method, sol_opt, prune_type):
    
    w1 = np.copy(w1)
    n, p = X.shape
    sparsity = 1-k/len(w1)
    if block_diag == None:
        block_diag = [0,p]
        
    beta = deter_groupsupp(w1, k, model, modules_to_prune, prune_type)
    
    #print("sp is ",len(np.where(beta)[0]))
    #print("sp is ",prune_list)
    #print(k,sparsity)
    if solve_method == "MP":
        w_pruned = np.copy(beta)
            
    elif solve_method == "WF":
            
        w_advpre = np.copy(w1)
        w_pruned = np.zeros(p)
        for jj in range(len(block_diag)-1):
            w_pruned[block_diag[jj]:block_diag[jj+1]] = WF_solve(X[:,block_diag[jj]:block_diag[jj+1]], 
                                beta[block_diag[jj]:block_diag[jj+1]], w_advpre[block_diag[jj]:block_diag[jj+1]], lambda2)
        
    elif solve_method == "BS":    
                 
        w_pruned, _, _, sol_time = L0_card.Heuristic_LSBlock(np.copy(w1),
                X,beta,len(np.where(beta)[0]),alpha=alpha, lambda1=0.,lambda2=lambda2, beta_tilde1=np.zeros(p), 
                beta_tilde2=np.copy(w1), M=np.inf,use_prune = False, per_idx=None, num_block=None, 
                block_list=block_diag, split_type=1)
        
    
    obj = 0.5 * np.linalg.norm(y-X@w_pruned)**2 + lambda2 *  np.linalg.norm(w1-w_pruned)**2 
    
    return w_pruned, obj