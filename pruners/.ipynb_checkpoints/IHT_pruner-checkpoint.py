from .utils import *
import time
from pyhessian.hessian import hessian #Needed for experiments with scaling the first order term

class IHTCDLSPruner:

    def __init__(self,model,params,prun_dataloader,
    ngrads,fisher_mini_bsz,criterion,blocksize,lambda2,num_iterations,
    first_order_term,compute_trace_H,alpha_one,
    device,algo='Active_IHTCDLS'):
        '''
         This object changes the model. 
        After prune is called, the attribute results is filled with the following keys :
        'norm_w_wbar','sparsity','new_non_zeros','trace_C','trace_H',
        'gradient_norm','obj','prun_runtime','norm_w'
        '''
        self.model = model
        self.params = params 
        self.prun_dataloader = prun_dataloader
        self.criterion = criterion
        self.ngrads = ngrads
        self.blocksize = blocksize
        self.lambda2 = lambda2*ngrads/2 #self.lambda2 is the lambda in the regression formulation
        self.num_iterations = num_iterations 
        self.device = device 
        self.first_order_term =first_order_term
        self.compute_trace_H  = compute_trace_H
        self.alpha_one = alpha_one
        self.fisher_mini_bsz = fisher_mini_bsz
        self.algo = algo
        self.grads = None
        self.results = dict()

        if self.blocksize > 0:
            self.algo = 'Heuristic_LSBlock' #This is the only block IHT algorithm
            self.block_list = get_blocklist(self.model,self.params,self.blocksize)

    def update_model(self,new_w):
        set_pvec(new_w, self.model,self.params,self.device)

    def reset_pruner(self):
        self.results = dict()
        self.grads=None
        
    def prune(self,mask,sparsity,grads=None):
        original_weight = get_pvec(self.model, self.params)
        if mask is None:
            mask = torch.ones_like(original_weight).cpu() != 0
        w1 = original_weight.to('cpu').numpy()
        d = len(w1)
        k = int((1-sparsity)*original_weight.numel())

        zero_grads(self.model)
        self.model.eval()


        if grads is None and self.grads is None:
            grads = torch.zeros((self.ngrads, d), device='cpu')
            for i, batch in enumerate(self.prun_dataloader):
                if i%100 ==0:
                    print('Computing gradients',i)
                if (i + 1) % self.ngrads == 0:
                    break
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                loss = self.criterion(self.model(x), y)
                loss.backward()
                grads[i] = get_gvec(self.model, self.params).to('cpu')
                zero_grads(self.model)
            grads = grads.numpy()
            if self.algo != 'Heuristic_LSBlock' and self.algo != 'Heuristic_LS':
                grads = grads.astype(np.float64)
        self.grads = grads
        
        y=grads@w1
        beta_tilde2=np.copy(w1)
        trace_C = np.linalg.norm(grads,ord='fro')**2/self.ngrads
        beta_tilde1 = np.zeros_like(w1)


        ###########################
        if self.first_order_term:
            if self.alpha_one and not self.compute_trace_H:
                trace_H = None
                alpha = 1
            elif not self.compute_trace_H: ##Don't compute trace(H)
                trace_H = None
                alpha = 1/self.fisher_mini_bsz
            else:
                start_hessian = time.time()
                for inputs_trace,outputs_trace in self.prun_dataloader:
                    break
                hessian_comp = hessian.hessian(self.model, self.criterion, data=(inputs_trace,outputs_trace),max_size=self.fisher_subsample_size, device=self.device)
                trace = hessian_comp.trace()
                trace_H = np.mean(trace)
                print('Hessian computation took', time.time() - start_hessian)
                alpha = (trace_C/trace_H)
                print('alpha =',alpha)

        else:
            trace_H = None 
            alpha = 0
        ###########################
        
        if alpha != 0:
            alpha_vec = alpha*grads.sum(axis=0) 
        else:
            alpha_vec = np.zeros_like(w1)

        gradient_norm = np.linalg.norm(grads.T@np.ones(self.ngrads), ord=2) / self.ngrads

        print('Starting Optimization')
        
        start_algo = time.time()

        if self.algo == 'Active_IHTCDLS':
            w_pruned, obj, _, _, _, sol_time = Active_IHTCDLS_PP(y,grads,w1,k,alpha=alpha_vec,lambda1=0,lambda2=self.lambda2,M=np.inf, beta_tilde1=beta_tilde1,
                        beta_tilde2=beta_tilde2, L=None, iht_max_itr=self.num_iterations, ftol = 1e-7, act_max_itr=self.num_iterations, buget=None, kimp=1.5, act_itr=1,
                        cd_itr = 0, ctol = 1e-4, sea1_max_itr=5, sea2_max_itr=10)
        elif self.algo == 'Heuristic_LS':
            w_pruned, obj, _, sol_time = Heuristic_LS(y,grads,w1,k,alpha=alpha_vec,lambda1=0,lambda2=self.lambda2,M=np.inf, beta_tilde1=beta_tilde1, 
                    beta_tilde2=beta_tilde2, use_prune=True)
        elif self.algo=='Heuristic_CD':
             w_pruned, obj, _, sol_time = Heuristic_CD_PP(y,grads,w1,k,alpha=alpha_vec,lambda1=0,lambda2=self.lambda2,M=np.inf, beta_tilde1=beta_tilde1, 
                        beta_tilde2=beta_tilde2, cd_max_itr=self.num_iterations, buget=None, kimp=1,sto_m='cyc',cd_tol=-1)
        elif self.algo == 'Heuristic_LSBlock':
            w_pruned, obj, _, sol_time = Heuristic_LSBlock(w1,grads,w1,k,alpha=alpha_vec,lambda1=0,lambda2=self.lambda2,M=np.inf, beta_tilde1=beta_tilde1, 
                        beta_tilde2=beta_tilde2, use_prune=True,block_list=self.block_list, split_type=1)

        end_algo = time.time()

        #set_pvec(w_pruned, self.model,self.params,self.device)

        self.results['trace_C'] = (trace_C)
        self.results['trace_H']=(trace_H)
        self.results['gradient_norm']=(gradient_norm)
        self.results['norm_w_wbar']=(np.linalg.norm(w_pruned-w1,ord=2))
        self.results['sparsity']=(sparsity)
        new_nz = (w_pruned[w1 == 0] != 0).sum()
        self.results['new_non_zeros']=(new_nz)
        self.results['obj']=(obj)
        self.results['prun_runtime']=(end_algo - start_algo)
        self.results['norm_w']=(np.linalg.norm(w_pruned,ord=2))
        #self.results['test_acc']=(compute_acc(self.model,self.test_dataloader,self.device))


        new_mask = torch.from_numpy(w_pruned != 0)

        

        return w_pruned,new_mask
        
