import torch
from time import time
from collections import namedtuple

USE_SVD=False

def mvm(A,b,index,transpose=False):
    n,p = A.shape
    if not transpose:
        res = torch.zeros(n,dtype=b.dtype)
        for j,i in enumerate(index):
            res += A[:,i]*b[j]
    else:
        res = torch.zeros(len(index),dtype=b.dtype)
        for j,i in enumerate(index):
            res[j] = A[:,i]@b
    return res


def mmm(A,index):
    n,p = A.shape
    res = torch.zeros((n,n),dtype=A.dtype)
    for i in range(n):
        res[i] = mvm(A,A[i][index],index,False)

    return res



def prune_ls(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M=torch.inf,sea_max_itr=5):
    
    _, p = X.shape
    device = X.device
    argsort = torch.argsort(-torch.abs(beta))
    support = argsort[:k]
    support_inv = argsort[k:]
    
    XTr = X.T@r
    beta_sub2 = beta_tilde2 - beta
    grad = -XTr + alpha - 2*lambda2*beta_sub2
    grad = grad.type(X.dtype)
    grad_supp = torch.zeros(p,dtype=X.dtype,device=device)
    grad_supp[support] = grad[support]
    
    if p > 5*k:
        Xgrad = X[:,support]@grad[support]
    else:
        Xgrad = X@grad_supp
        
    opt_step = ((grad_supp@alpha) - (r@Xgrad) - 2*lambda2*(beta_sub2@grad_supp))/((Xgrad@Xgrad) +2*lambda2*(grad_supp@grad_supp))
        
    sup_max = torch.max(torch.abs(beta[support]-opt_step*grad[support]))
    supinv_max = torch.max(torch.abs(beta[support_inv]-opt_step*grad[support_inv]))
    
    if sup_max >= supinv_max - 1e-10:
        # opt_step is less change step
        #print("Use opt step",opt_step)
        beta_new = beta - opt_step*grad
        beta_new = beta_new.type(X.dtype)
        beta_new[argsort[k:]] = 0 
        if p > 5*k:
            r = y - X[:,argsort[:k]]@beta_new[argsort[:k]]
        else:
            r = y - X@beta_new
        L_best = opt_step
    else:
        #print("Use line search step, opt step is",opt_step)
        beta_new=beta
        L_step = opt_step/2
        L_best = L_step
        sea_itr = 0
        while sea_itr < 100:
            sup_max = torch.max(torch.abs(beta[support]-L_step*grad[support]))
            supinv_max = torch.max(torch.abs(beta[support_inv]-L_step*grad[support_inv]))
            if sup_max >= supinv_max - 1e-10:
                break
            L_step /= 2
            sea_itr += 1
        
        beta_mp = torch.zeros(p,device=device)
        beta_mp[support] = beta[support]
        beta_mp = beta_mp.type(X.dtype)
        f_best = evaluate_obj(beta_mp,y-X@beta_mp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2) 
        sea_itr = 0
        while sea_itr < sea_max_itr:
            
            beta_tmp = beta - L_step*grad
            beta_tmp = beta_tmp.type(X.dtype)
            argsort = torch.argsort(-torch.abs(beta_tmp))
            beta_tmp[argsort[k:]] = 0
            if p > 5*k:
                r_tmp = y - X[:,argsort[:k]]@beta_tmp[argsort[:k]]
            else:
                r_tmp = y - X@beta_tmp
                
            f_new = evaluate_obj(beta_tmp,r_tmp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
            #print("f_new is",f_new,"f_best is",f_best,"step is", L_step)
            if f_new < f_best:
                f_best = f_new
                beta_new = torch.clone(beta_tmp)
                r = torch.clone(r_tmp)
                L_best = L_step
            else:
                break
            L_step *= 2
            sea_itr += 1
            
    return beta_new, r,L_best


def evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2):
    beta_sub1 = beta - beta_tilde1
    beta_sub2 = beta - beta_tilde2
    return 0.5*(r@r) + lambda2*(beta_sub2@beta_sub2) + lambda1*(torch.sum(torch.abs(beta_sub1))) + alpha@beta



def compute_inverse(y, X, alpha, lambda2, beta_tilde2,act_idx=None):
    n, p = X.shape
    k = p if act_idx is None else len(act_idx)
    device = X.device
    if n < k:
        if act_idx is None:
            solve_b = X.T@y + 2*lambda2*beta_tilde2 - alpha
            beta = solve_b / (2*lambda2) - \
            X.T@torch.linalg.solve(torch.eye(n,device=device)+X@(X.T)/(2*lambda2),X@solve_b)/(4*lambda2**2)
        else:
            solve_b = mvm(X,y,act_idx,True) + 2*lambda2*beta_tilde2[act_idx] - alpha[act_idx]
            mmX = mmm(X,act_idx)
            mvmX =mvm(X,solve_b,act_idx,False)
            solve_tmp = (torch.linalg.solve(torch.eye(n,device=device)+mmX/(2*lambda2),mvmX)/(4*lambda2**2)).astype(X.dtype)
            beta = solve_b / (2*lambda2) - \
                mvm(X,solve_tmp,act_idx,True)
    else:
        if act_idx is None:
            solve_b = X.T@y + 2*lambda2*beta_tilde2 - alpha
            beta = torch.linalg.solve(2*lambda2*torch.eye(k,device=device)+(X.T)@X,solve_b)
        else:
            solve_b = mvm(X,y,act_idx,True) + 2*lambda2*beta_tilde2[act_idx] - alpha[act_idx]
            X_act = X[:,act_idx]
            beta = torch.linalg.solve(2*lambda2*torch.eye(k,device=device)+(X_act.T)@X_act,solve_b)
    return beta.type(X.dtype)

def Heuristic_LS(y,X,beta,k,alpha,lambda1,lambda2, beta_tilde1, beta_tilde2, M=torch.inf,
                use_prune = True):
    
    
    st = time()
    n, p = X.shape
    device = X.device
    alpha = torch.zeros(p,device=device) if alpha is None else alpha
    beta_tilde1 = torch.zeros(p,device=device) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = torch.zeros(p,device=device) if beta_tilde2 is None else beta_tilde2
    if X.dtype == 'float32':
        lambda2 = torch.float32(lambda2,device=device)
        lambda1 = torch.float32(lambda1,device=device)
    num_nnz = (beta != 0).sum()
    if num_nnz > k and torch.linalg.norm(alpha) > 1e-8 and use_prune:
        beta, r, L_best = prune_ls(y,X,beta,y-X@beta,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,10)
    active_set = torch.argsort(-torch.abs(beta))
    act_idx = active_set[:k]
    if p < 1e7:
        beta_act = beta[act_idx]
        X_act = X[:,act_idx]
        beta_act = compute_inverse(y, X_act, alpha[act_idx], lambda2, beta_tilde2[act_idx])
    else:
        beta_act = compute_inverse(y, X, alpha, lambda2, beta_tilde2,act_idx)
    
    beta = torch.zeros(p,dtype=X.dtype,device=device)
    beta[act_idx] = beta_act
    r = y - X@beta
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
    sol_time = time()-st
    
    return beta, f, r, sol_time

def Heuristic_LSBlock(w_bar,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, M=torch.inf,
                use_prune = True, per_idx=None, num_block=1, block_list=None, split_type=0):
    
    
    st = time()
    n, p = X.shape
    device = X.device
    alpha = torch.zeros(p,device=device) if alpha is None else alpha
    beta_tilde1 = torch.zeros(p,device=device) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = torch.zeros(p,device=device) if beta_tilde2 is None else beta_tilde2
    if block_list is None:
        block_list = list(range(0,p+1,int(p/num_block))) 
        block_list[-1] = p

    if X.dtype == 'float32':
        lambda2 = torch.float32(lambda2,device=device)
        lambda1 = torch.float32(lambda1,device=device)
    
    y = X@w_bar
    X_per = X
    w_barper = w_bar
    beta_per = beta
    alpha_per = alpha
    beta_tilde1per = beta_tilde1
    beta_tilde2per = beta_tilde2
    
    beta_new = torch.zeros(p,dtype=X.dtype,device=device)
    ksum = 0
    
    if torch.linalg.norm(alpha) > 1e-8 and use_prune:
        beta, r = prune_ls(y,X,beta,y-X@beta,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,10)
        beta_per = beta

    active_set = torch.argsort(-torch.abs(beta))
    thres = torch.abs(beta[active_set[k]])
    
    for ib in range(len(block_list)-1):
        idx_cur = torch.arange(block_list[ib],block_list[ib+1])
        kcur = torch.sum(torch.abs(beta_per[idx_cur]) > thres)
        ksum += kcur
        if kcur == 0:
            continue
        beta_new[idx_cur], _, _, _ = Heuristic_LS(y,X_per[:,idx_cur],beta_per[idx_cur],kcur,alpha_per[idx_cur],
            lambda1,lambda2,beta_tilde1per[idx_cur],beta_tilde2per[idx_cur],M,use_prune)
    
    sol_time = time() - st
    r = y-X@beta_new
    f = evaluate_obj(beta_new,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
    
    #print("Total non-zero:",ksum)
    return beta_new, f, r, sol_time

def prox_L1(beta, beta_tilde1, lambda1):
    beta_sub = beta-beta_tilde1
    abs_beta = torch.abs(beta_sub)
    return beta_tilde1 + torch.where(abs_beta>lambda1, abs_beta-lambda1, torch.zeros_like(abs_beta-lambda1)) * torch.sign(beta_sub)

def clip(beta, M):
    device = beta.device
    abs_beta = torch.abs(beta)
    return torch.where(abs_beta>M, torch.tensor([M],device=device).type(beta.type())[0] , abs_beta) * torch.sign(beta)

def hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M=torch.inf):
    
    beta_hat = (1-2*lambda2/L)*beta + (X.T@r-alpha+2*lambda2*beta_tilde2)/L
    beta_new = clip(prox_L1(beta_hat, beta_tilde1, lambda1/L), M)
    rec_obj = (1/2)*(beta_new - beta_hat)**2+lambda1/L*torch.abs(beta_new-beta_tilde1) - ((1/2)*(beta_hat)**2+lambda1/L*torch.abs(beta_tilde1))
    argsort = torch.argsort(rec_obj)
    beta_new[argsort[k:]] = 0
    
    _, p = X.shape
    if p > 5*k:
        r = y - X[:,argsort[:k]]@beta_new[argsort[:k]]
    else:
        r = y - X@beta_new
    
    return beta_new, r

def coordinate_descent(y,X,beta,r,i,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M=torch.inf):

    beta_inew = clip(prox_L1(2*lambda2*beta_tilde2[i]+X[:,i]@r+S_diag[i]*beta[i]-alpha[i], beta_tilde1[i]*(S_diag[i]+2*lambda2), lambda1)/(S_diag[i]+2*lambda2), M)
    
    return beta_inew


def CD_loop(y,X,beta,r,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M=torch.inf, cd_itr=1, sto_m = "cyc", cd_tol = -1):
    
    f_old = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
    for i in range(cd_itr):
        support = torch.where(beta!=0)[0]
        if sto_m == "cyc":
            for j in support:
                beta_inew = coordinate_descent(y,X,beta,r,j,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M)
                r = r - (beta_inew - beta[j])*X[:,j]
                beta[j] = beta_inew
        elif sto_m == "sto":
            for _ in range(len(support)):
                j = np.random.randint(len(support))
                beta_inew = coordinate_descent(y,X,beta,r,j,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M)
                r = r - (beta_inew - beta[j])*X[:,j]
                beta[j] = beta_inew
        
        f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
        if (abs(f_old - f) <= max(abs(f),abs(f_old),1)* cd_tol):
            break
        f_old = f
                        
            
    return beta, r

def Vanilla_IHT(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0.,beta_tilde1=None,beta_tilde2=None,L=None,M=torch.inf,iht_max_itr=100,ftol=1e-8):
    
    assert lambda1==0
    assert M==torch.inf
    
    st = time()
    p = beta.shape[0]
    device = X.device
    alpha = torch.zeros(p,device=device) if alpha is None else alpha
    beta_tilde1 = torch.zeros(p,device=device) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = torch.zeros(p,device=device) if beta_tilde2 is None else beta_tilde2
    L = 1.05*(skl_svd(X,use_svd=USE_SVD)**2+lambda2*2) if L is None else L
    #print('L--',L)
    r = y - X@beta
    S_diag = torch.linalg.norm(X, axis=0)**2
    iht_cur_itr = 0
    objs = []
    beta, r = hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M)
    f_old = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
    while iht_cur_itr < iht_max_itr:
        
        beta, r = hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M)
        f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
        #if (abs(f_old - f) <= max(abs(f),abs(f_old),1)* ctol) and iht_cur_itr > 0:
        #    beta,r = CD_loop(y,X,beta,r,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,cd_itr)
        #    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
                
        objs.append(f)
        if (abs(f_old - f) <= max(abs(f),abs(f_old),1)*ftol) and iht_cur_itr > 0:
            break
        f_old = f
        iht_cur_itr += 1
        
    sol_time = time()-st
    return beta, f, objs, r, iht_cur_itr, sol_time
                        

def Vanilla_IHTCDLS(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0.,beta_tilde1=None,beta_tilde2=None,L=None,M=torch.inf,iht_max_itr=100,ftol=1e-8,
                  cd_itr=0,ctol=1e-4,search_max_itr=1):
    
    assert lambda1==0
    assert M==torch.inf
    
    st = time()
    p = beta.shape[0]
    device = X.device
    alpha = torch.zeros(p,device=device) if alpha is None else alpha
    beta_tilde1 = torch.zeros(p,device=device) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = torch.zeros(p,device=device) if beta_tilde2 is None else beta_tilde2
    L = 1.05*(skl_svd(X,use_svd=USE_SVD)**2+lambda2*2) if L is None else L
    r = y - X@beta
    S_diag = torch.linalg.norm(X, axis=0)**2
    iht_cur_itr = 0
    objs = []
    Ls = []
    beta, r = hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M)
    f_old = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
    while iht_cur_itr < iht_max_itr:
        
        beta, r,L_best = hard_thresholding_ls(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M, search_max_itr)
        f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
        #if (abs(f_old - f) <= max(abs(f),abs(f_old),1)* ctol) and iht_cur_itr > 0:
        #    beta,r = CD_loop(y,X,beta,r,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,cd_itr)
        #    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
                
        objs.append(f)
        Ls.append(L_best)
        if (abs(f_old - f) <= max(abs(f),abs(f_old),1)*ftol) and iht_cur_itr > 0:
            break
        f_old = f
        iht_cur_itr += 1
        
    sol_time = time()-st
    return beta, f, objs,Ls, r, iht_cur_itr, sol_time


def Active_IHTCDLS(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, L=None, M=torch.inf, 
                iht_max_itr=100, ftol=1e-8, act_max_itr=10,buget=None,kimp=2.,act_itr=1,cd_itr=0,ctol=1e-4,
                sea1_max_itr=5, sea2_max_itr=10):

    
    st = time()
    p = beta.shape[0]
    device = X.device
    alpha = torch.zeros(p,device=device) if alpha is None else alpha
    beta_tilde1 = torch.zeros(p,device=device) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = torch.zeros(p,device=device) if beta_tilde2 is None else beta_tilde2
    L = 1.05*(skl_svd(X,use_svd=USE_SVD)**2+lambda2*2) if L is None else L
    r = y - X@beta
    f_old = torch.inf
    act_cur_itr = 0
    _sol_str = 'active_set objs Ls, r, iht_cur_itr sol_time search_itr outliers'
    Solution = namedtuple('Solution', _sol_str)
    sols = []
    
    active_set = initial_active_set(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M,buget,kimp,act_itr)
    
    while act_cur_itr < act_max_itr:
        
        X_act = X[:,active_set]
        beta_act = beta[active_set]
        L_act = 1.05*(skl_svd(X_act,use_svd=USE_SVD)**2+lambda2*2) 
        beta_act, f, objs,Ls, r_act, iht_cur_itr, sol_time = Vanilla_IHTCDLS(y,X_act,beta_act,k,alpha[active_set],lambda1,lambda2,beta_tilde1[active_set],
                                                                     beta_tilde2[active_set], L_act,M,iht_max_itr,ftol,cd_itr,ctol,sea1_max_itr)
        
        #print("Num of iter:",act_cur_itr+1," num of inner iter:",iht_cur_itr,"\n Finding new active set")
        L_init = 2*L_act
        beta = torch.zeros(p,device=device)
        beta[active_set] = beta_act
        r = y - X@beta
        f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
        active_set_set = set(active_set)
        search_flag = False
        search_cur_itr = 0
        outliers = set()
        beta_update,r_update = beta,r
        while search_cur_itr < sea2_max_itr:
            beta_tmp, r_tmp = hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L_init,M)
            f_new = evaluate_obj(beta_tmp,r_tmp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
            outliers = set(torch.where(beta_tmp)[0].cpu().numpy()) - active_set_set
            search_cur_itr += 1
            #print(f_new.item(),f.item(),len(outliers),search_cur_itr)
            if len(outliers) >= 1 and f_new < f:
                search_flag = True
                beta_update = beta_tmp
                r_update = r_tmp
            elif f_new >= f:
                beta = beta_update 
                r = r_update
                break
            L_init /= 2
            
        sols.append(Solution(active_set=active_set,objs=objs,Ls=Ls,r=torch.clone(r_act),iht_cur_itr=iht_cur_itr,sol_time=sol_time,
                             search_itr=search_cur_itr,outliers=len(outliers)))
        if not search_flag:
            break
        active_set = list(sorted(active_set_set | outliers))
        act_cur_itr += 1
    
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
    tot_time = time()-st
    return beta, f, sols, r, act_cur_itr, tot_time

def initial_active_set(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M=torch.inf,buget=None,kimp=2.,act_itr=1):
    
    p = beta.shape[0]
    device = beta.device
    buget = p if buget is None else buget
    ksupp = int(torch.max(torch.tensor([torch.min(torch.Tensor([kimp*k, buget, p])),k],device=device)))
    beta_tmp, r_tmp = torch.clone(beta), torch.clone(r)
    for i in range(act_itr):
        beta_tmp,r_tmp = hard_thresholding(y,X,beta_tmp,r_tmp,ksupp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M)
    active_set = set(torch.where(beta_tmp)[0].cpu().numpy())    
    active_set = list(sorted(active_set))
    
    return active_set

def hard_thresholding_ls(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M=torch.inf,sea_max_itr=5):
    
    _, p = X.shape
    device = X.device
    support = torch.where(beta!=0)[0]
    support_inv = torch.where(beta==0)[0]
    XTr = X.T@r
    beta_sub2 = beta_tilde2 - beta
    grad = -XTr + alpha - 2*lambda2*beta_sub2
    grad_supp = torch.zeros(p,device=device)
    grad_supp[support] = grad[support]
    
    max_suppinv = torch.max(torch.abs(grad[support_inv]))
    same_sign = torch.sign(beta[support])==torch.sign(grad[support])
    L_change = torch.min(torch.where( same_sign + (torch.abs(grad[support])<max_suppinv),
         torch.abs(beta[support])/(torch.abs(grad[support])*(2*same_sign-1)+max_suppinv),torch.tensor([torch.inf],device=device).type(X.type())))
   
    
    if p > 5*k:
        Xgrad = X[:,support]@grad[support]
    else:
        Xgrad = X@grad_supp
    
 
    opt_step = (grad_supp@alpha - r@Xgrad - 2*lambda2*beta_sub2@grad_supp)/(Xgrad@Xgrad+2*lambda2*grad_supp@grad_supp)
    
    #print("----opt step is",opt_step,"Lchange is",L_change)
    if opt_step < L_change:
        L_best = opt_step
        beta_new = beta - opt_step*grad
        argsort = torch.argsort(-torch.abs(beta_new))
        beta_new[argsort[k:]] = 0
        if p > 5*k:
            r = y - X[:,argsort[:k]]@beta_new[argsort[:k]]
        else:
            r = y - X@beta_new
    else:
        f_best = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)   
        sea_itr = 0
        #L_step = np.maximum(L_change,1/L)
        L_step = L_change/(1+1e-4)
        L_best = L_step
        beta_new = torch.clone(beta)
        while sea_itr < sea_max_itr:
            
            beta_tmp = beta - L_step*grad
            argsort = torch.argsort(-torch.abs(beta_tmp))
            beta_tmp[argsort[k:]] = 0
            if p > 5*k:
                r_tmp = y - X[:,argsort[:k]]@beta_tmp[argsort[:k]]
            else:
                r_tmp = y - X@beta_tmp
                
            f_new = evaluate_obj(beta_tmp,r_tmp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
            #print("f_new is",f_new,"f_best is",f_best,"step is",L_step)
            if f_new < f_best:
                f_best = f_new
                beta_new = torch.clone(beta_tmp)
                r = torch.clone(r_tmp)
                L_best = L_step
            else:
                break
         
            L_step *= 2
            sea_itr += 1
    #print('Best L',L_best)
    return beta_new, r,L_best

def skl_svd(X,use_svd=True):
    #return extmath.randomized_svd(X,n_components=1)[1][0]
    if use_svd:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        return S[0].item()
    else:
        return torch.linalg.matrix_norm(X,ord='fro').item()


def IHT_LSBlock(w_bar,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, M=torch.inf,
                use_prune = True, per_idx=None, num_block=1, block_list=None, num_iterations=0):
    
    
    st = time()
    n, p = X.shape
    device = X.device
    alpha = torch.zeros(p,device=device) if alpha is None else alpha
    beta_tilde1 = torch.zeros(p,device=device) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = torch.zeros(p,device=device) if beta_tilde2 is None else beta_tilde2
    if block_list is None:
        block_list = list(range(0,p+1,int(p/num_block))) 
        block_list[-1] = p

    if X.dtype == 'float32':
        lambda2 = torch.float32(lambda2,device=device)
        lambda1 = torch.float32(lambda1,device=device)
    
    y = X@w_bar
    X_per = X
    w_barper = w_bar
    beta_per = beta
    alpha_per = alpha
    beta_tilde1per = beta_tilde1
    beta_tilde2per = beta_tilde2
    
    beta_new = torch.zeros(p,dtype=X.dtype,device=device)
    ksum = 0
    
    if torch.linalg.norm(alpha) > 1e-8 and use_prune:
        beta, r = prune_ls(y,X,beta,y-X@beta,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,10)
        beta_per = beta

    active_set = torch.argsort(-torch.abs(beta))
    thres = torch.abs(beta[active_set[k]])
    
    for ib in range(len(block_list)-1):
        idx_cur = torch.arange(block_list[ib],block_list[ib+1])
        kcur = torch.sum(torch.abs(beta_per[idx_cur]) > thres)
        ksum += kcur
        if kcur == 0:
            continue
        beta_new[idx_cur], _, _, _ ,_,_= Active_IHTCDLS(X_per[:,idx_cur]@w_barper[idx_cur],X_per[:,idx_cur],beta_per[idx_cur],kcur,alpha_per[idx_cur],
            lambda1,lambda2,beta_tilde1per[idx_cur],beta_tilde2per[idx_cur],M=M,iht_max_itr=num_iterations)
    
    sol_time = time() - st
    r = y-X@beta_new
    f = evaluate_obj(beta_new,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
    
    #print("Total non-zero:",ksum)
    return beta_new, f, r, sol_time

def Vanilla_IHTCDLS_PP(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, L=None, M=torch.inf, iht_max_itr=100, ftol=1e-8, act_max_itr=10,buget=None,kimp=2.,act_itr=1,cd_itr=0,ctol=1e-4,
  sea_max_itr=5):
    
    
    nnz_idx = torch.where(torch.linalg.norm(X, axis=0)**2)[0]
    beta_new = torch.zeros_like(beta)
    
    if len(nnz_idx) > k:
        beta, f, sols,Ls, r, act_cur_itr, tot_time = Vanilla_IHTCDLS(y,X[:,nnz_idx],beta[nnz_idx], min(k,len(nnz_idx)),alpha=alpha[nnz_idx],lambda1=lambda1,lambda2=lambda2,
                    beta_tilde1=beta_tilde1[nnz_idx], beta_tilde2=beta_tilde2[nnz_idx], L=L, M=M, iht_max_itr=iht_max_itr, ftol=ftol,       
                    cd_itr=cd_itr,ctol=ctol,search_max_itr=sea_max_itr)
                    
        
        beta_new[nnz_idx] = torch.clone(beta)
    else:
        beta, f, sols, r, act_cur_itr, tot_time = Vanilla_IHT(y,X[:,nnz_idx],beta[nnz_idx],min(k,len(nnz_idx)),alpha=alpha[nnz_idx],lambda1=lambda1,lambda2=lambda2,beta_tilde1=beta_tilde1[nnz_idx], beta_tilde2=beta_tilde2[nnz_idx], L=L, M=M, iht_max_itr=iht_max_itr, ftol=ftol)
        beta_new[nnz_idx] = torch.clone(beta)
    #else:
    #    beta_new[nnz_idx] = torch.clone(beta[nnz_idx])
    #    f = 0
    #    r = torch.zeros_like(y)
    #    sols = None
    #    act_cur_itr=0
    #    tot_time=0
    
    return beta_new, f, sols, r, act_cur_itr, tot_time


def Active_IHTCDLS_PP(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, L=None, M=torch.inf, iht_max_itr=100, ftol=1e-8, act_max_itr=10,buget=None,kimp=2.,act_itr=1,cd_itr=0,ctol=1e-4,
  sea1_max_itr=5, sea2_max_itr=10):
    
    
    nnz_idx = torch.where(torch.linalg.norm(X, axis=0)**2)[0]
    beta_new = torch.zeros_like(beta)
    
    if len(nnz_idx) > k:
        beta, f, sols, r, act_cur_itr, tot_time = Active_IHTCDLS(y,X[:,nnz_idx],beta[nnz_idx], min(k,len(nnz_idx)),alpha=alpha[nnz_idx],lambda1=lambda1,lambda2=lambda2,
                    beta_tilde1=beta_tilde1[nnz_idx], beta_tilde2=beta_tilde2[nnz_idx], L=L, M=M, iht_max_itr=iht_max_itr, ftol=ftol,       
                    act_max_itr=act_max_itr,buget=buget,kimp=kimp,act_itr=act_itr,cd_itr=cd_itr,ctol=ctol,
                    sea1_max_itr=sea1_max_itr, sea2_max_itr=sea2_max_itr)
        
        beta_new[nnz_idx] = torch.clone(beta)
    else:
        beta, f, sols, r, act_cur_itr, tot_time = Vanilla_IHT(y,X[:,nnz_idx],beta[nnz_idx],min(k,len(nnz_idx)),alpha=alpha[nnz_idx],lambda1=lambda1,lambda2=lambda2,beta_tilde1=beta_tilde1[nnz_idx], beta_tilde2=beta_tilde2[nnz_idx], L=L, M=M, iht_max_itr=iht_max_itr, ftol=ftol)
        beta_new[nnz_idx] = torch.clone(beta)
    #else:
    #    beta_new[nnz_idx] = torch.clone(beta[nnz_idx])
    #    f = 0
    #    r = torch.zeros_like(y)
    #    sols = None
    #    act_cur_itr=0
    #    tot_time=0
    
    return beta_new, f, sols, r, act_cur_itr, tot_time


def IHT_LSBlock_PP(y,X,beta,k,block_list,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, L=None, M=torch.inf, iht_max_itr=100, ftol=1e-8, act_max_itr=10,buget=None,kimp=2.,act_itr=1,cd_itr=0,ctol=1e-4,
  sea1_max_itr=5, sea2_max_itr=10):
    
    
    nnz_idx = torch.where(torch.linalg.norm(X, axis=0)**2)[0]
    beta_new = torch.zeros_like(beta)
    
    if True or len(nnz_idx) > k:
        beta, f, r, tot_time = IHT_LSBlock(beta[nnz_idx],X[:,nnz_idx],beta[nnz_idx],min(k,len(nnz_idx)),alpha=alpha[nnz_idx],lambda1=lambda1,lambda2=lambda2, beta_tilde1=beta_tilde1, beta_tilde2=beta_tilde2, M=M,
                use_prune = True,block_list=block_list, num_iterations=iht_max_itr)
        
        beta_new[nnz_idx] = torch.clone(beta)
    else:
        beta_new[nnz_idx] = torch.clone(beta[nnz_idx])
        f = 0
        r = torch.zeros_like(y)
        sols = None
        act_cur_itr=0
        tot_time=0
    
    return beta_new, f, r, tot_time