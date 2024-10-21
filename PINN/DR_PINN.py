'''
@Project ：Diffusion Reaction PDE
@File    ：DR_PINN.py
@IDE     ：VS Code
@Author  ：Cheng
@Date    ：2024-10-21
'''
from DNN import PINN
from DNN import Attention_PINN

import torch
import numpy as np
import timeit
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import trange
import random
import pandas as pd

"""Random Seed Setup"""
def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(123)

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

"""Data Driven - We need a good Sampler"""
class Sampler: ##Grid Sampler
    def __init__(self, geom, time, name = None):
        self.geom = geom  ##Length of the domain
        self.time = time  ##Annealing time
        self.name = name

    def generate_non_dimensional_data(self,num_interior,num_boundary,num_initial,additional_points = 0):
        """
        generate num_interior^2 domain points、num_boundar*2 boundary points、num_initial initial points.
        """
        x_start,x_end = self.geom[0],self.geom[1]
        t_start,t_end = self.time[0],self.time[1]
        
        ###Generate interior points
        x = torch.linspace(x_start,x_end/x_end,
                           steps= num_interior,device = device,
                           requires_grad=True)
        t = torch.linspace(t_start,t_end/t_end,
                           steps= num_interior,device = device,
                           requires_grad=True)

        ###Maybe we need some additional points in reacion-Zone, normally we set additional_points = 0 as default
        x_additional = torch.linspace(0/x_end,300/x_end,steps=additional_points,
                                      device=device,requires_grad=True)
        t_additional = torch.linspace(t_start,t_end/t_end,steps=additional_points,
                                      device=device,requires_grad=True)
        
        X_ad,T_ad = torch.meshgrid(x_additional, t_additional, indexing='ij')
        points_additional = torch.stack((X_ad.flatten(), T_ad.flatten()), dim=1).to(device)

        X,T = torch.meshgrid(x, t, indexing='ij')
        points_inter = torch.stack([X.flatten(), T.flatten()], dim = 1)
        points_inter = torch.cat((points_inter, points_additional), dim=0)

        indices_inter = torch.randperm(points_inter.size(0))
        points_inter = points_inter[indices_inter].to(device)

        ###Generate boundary points
        x_bound_left = torch.full((num_boundary,1),x_start)
        x_bound_right = torch.full((num_boundary,1),x_end/x_end)
        t_bound = torch.linspace(t_start,t_end/t_end,steps=num_boundary).unsqueeze(1)

        points_bound_left = torch.cat((x_bound_left, t_bound), dim=1)
        points_bound_right = torch.cat((x_bound_right, t_bound), dim=1)
        points_bound = torch.cat((points_bound_left,points_bound_right),dim=0)

        indices_bound = torch.randperm(points_bound.size(0))
        points_bound = points_bound[indices_bound].to(device)

        ###Generate initial points
        x_init = torch.linspace(0, x_end/x_end, num_initial).unsqueeze(1)
        t_init = torch.zeros(num_initial, 1)
        points_init = torch.cat((x_init, t_init), dim=1)
        indices_init = torch.randperm(points_init.size(0))
        points_init = points_init[indices_init].to(device)
        
        return points_inter, points_bound, points_init
    
"""Main Part of our Diffusion Reaction PDE Model and Training-Process"""
class Diffusion_Reaction:
    def __init__(self, layers,sampler,
                 D_A,k11,k12,k21,N_SA,N_SB,N_SC,N_SAB,N_SABB,N_SAAB,
                 h,L,anneal_time,
                 model_type,lam_threshold):
        
        ####Parameters of Neural Network and Sampler
        self.layers = layers
        self.sampler = sampler

        ####Parameters of Diffusion Reaction PDE
        self.D_A = D_A     ##Diffusivity
        self.k11 = k11     ##Reaction k11     
        self.k12 = k12     ##Reaction k12    
        self.k21 = k21     ##Reaction k21

        self.N_SA = N_SA     ##Characteristic concentration of Ni
        self.N_SB = N_SB     ##Characteristic concentration of SiC
        self.N_SC = N_SC     ##Characteristic concentration of C
        self.N_SAB = N_SAB    ##Characteristic concentration of NiSi
        self.N_SABB = N_SABB    ##Characteristic concentration of NiSi2
        self.N_SAAB = N_SAAB    ##Characteristic concentration of Ni2Si

        self.h = h          ##Thickness of Metal-Film h  
        self.L = L          ##Length of Reaction-Zone L
        self.anneal_time = anneal_time      ###Annealing time

        ####Model Type
        self.model_type = model_type
        self.lam_threshold = lam_threshold

        if self.model_type in ['PINN','IAW_PINN']:
            self.dnn = PINN(layers,use_layernorm=False).to(device)

        elif self.model_type in ['IA_PINN','I_PINN']:
            self.dnn = Attention_PINN(layers).to(device)
        
        """Here is the part of adaptive weights adjustment, but the effect is not ideal """
        self.var_ic1 = torch.tensor([1.], requires_grad=True).to(device)
        self.var_ic2 = torch.tensor([1.], requires_grad=True).to(device)

        self.var_ic1 = torch.nn.Parameter(self.var_ic1)
        self.var_ic2 = torch.nn.Parameter(self.var_ic2)

        self.optimizer_ic = torch.optim.Adam([self.var_ic1] + [self.var_ic2], lr=1e-3)  ##自适应权重的优化器

        self.lam_threshold = lam_threshold    ##Bound of loss weights

        '''Training Log'''
        self.iter = 0

        ##Log of losses
        self.loss_pde_log = []
        self.loss_bc_log = []
        self.loss_ic_log =[]
        self.loss_total_log =[]

        ##Log of penalties
        self.lam_ic1_log = []
        self.lam_ic2_log = []
        self.lam_ic_log = []

        ##Log of relative errors
        self.log_relative_u1 = []
        self.log_relative_u2 = []
        self.log_relative_u3 = []
        self.log_relative_u4 = []
        self.log_relative_u5 = []
        self.log_relative_u6 = []

        '''Gradient Function'''
    def gradients(self,u,x,order =1):
        """
        To calculate the gradient of function u with respect to variable x, higher-order derivatives can be calculated.
        Args:
            u (torch.Tensor): input function u.
            x (torch.Tensor): input variable x.
             order (int, optional): caculate the order of derivative. Defaults to 1.
        Returns:
            torch.Tensor: the gradient of u with respect to x.
        """
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                        create_graph=True,
                                        only_inputs=True )[0]
        else:
            return self.gradients(self.gradients(u, x), x, order=order - 1)
            
    '''The most important Part - Diffusion Reaction Equations'''
    def diffusion_reaction(self,coords_pde,coords_bc,coords_ic):
        '''Governing Equation Part'''
        coords_pde =coords_pde.clone().detach().requires_grad_(True)
        predictions = self.dnn(coords_pde)
        c_a,c_bc,c_c,c_ab,c_abb,c_aab = predictions[:,0],predictions[:,1],predictions[:,2],predictions[:,3],predictions[:,4],predictions[:,5]

        ##Derivative Terms
        gradients_t = [self.gradients(c, coords_pde)[:,1] for c in [c_a, c_bc, c_c, c_ab, c_abb, c_aab]]
        dca_dt,dcbc_dt,dcc_dt,dcab_dt,dcabb_dt,dcaab_dt = gradients_t   

        gradients_x = [self.gradients(c, coords_pde)[:,0] for c in [c_a, c_bc, c_c, c_ab, c_abb, c_aab]]
        dca_dx, dcbc_dx, dcc_dx, dcab_dx, dcabb_dx, dcaab_dx = gradients_x

        ##Construcion of compouned Diffusivity
        sig_numerator = c_bc + c_c + c_ab + c_abb + c_aab
        sig_denominator = c_a + c_bc + c_c + c_ab + c_abb + c_aab
        sig_denominator = torch.where((sig_denominator == 0), 1e-8, sig_denominator)
        sig_D = sig_numerator /sig_denominator
        D_A = self.D_A* (self.anneal_time/(self.L**2))  #With Non-Dimensionalization
        D_star = D_A * sig_D

        ##Residuals of Governing Equations with Non-Dimensionalization
        eq_1 = dca_dt - self.gradients(D_star*dca_dx,coords_pde)[:,0]  +\
            self.k11 *(self.N_SA * self.N_SB* self.anneal_time) / self.N_SA* c_a * c_bc + self.k21*(self.N_SA * self.N_SAB *self.anneal_time)/self.N_SA * c_a * c_ab
        
        eq_2 = dcbc_dt -self.gradients(D_star*dcbc_dx,coords_pde)[:,0]+\
            self.k11*(self.N_SA * self.N_SB*self.anneal_time)/ self.N_SB * c_a * c_bc + self.k12*(self.N_SAB * self.N_SB*self.anneal_time) /self.N_SB * c_ab * c_bc
        
        eq_3 = dcc_dt - self.gradients(D_star*dcc_dx,coords_pde)[:,0] -\
            self.k11*(self.N_SA* self.N_SB *self.anneal_time) /self.N_SC * c_a * c_bc - self.k12 *(self.N_SAB * self.N_SB*self.anneal_time) / self.N_SC * c_ab * c_bc    
            
        eq_4 = dcab_dt - self.k11*(self.N_SA * self.N_SB *self.anneal_time) / self.N_SAB *c_a *c_bc + self.k21 *(self.N_SA * self.N_SAB *self.anneal_time)/self.N_SAB* c_a * c_ab +\
            self.k12*(self.N_SAB * self.N_SB*self.anneal_time) /self.N_SAB * c_ab * c_bc
        
        eq_5 = dcabb_dt - self.k12 *(self.N_SAB * self.N_SB *self.anneal_time) / self.N_SABB * c_ab * c_bc
        
        eq_6 = dcaab_dt - self.k21 *(self.N_SA * self.N_SAB * self.anneal_time)/ self.N_SAAB * c_a * c_ab   

        loss_pde = torch.mean(eq_1**2) + torch.mean(eq_2**2) + torch.mean(eq_3**2) +\
            torch.mean(eq_4**2) + torch.mean(eq_5**2) + torch.mean(eq_6**2)

        '''Boundary Condition Part'''
        coords_bc = coords_bc.clone().detach().requires_grad_(True)
        predictions_bc = self.dnn(coords_bc)
        bc_ca,bc_cbc,bc_cc = predictions_bc[:,0],predictions_bc[:,1],predictions_bc[:,2]

        ##Neumann Boundary Condition
        bc_ca_dx = self.gradients(bc_ca, coords_bc)[:, 0]
        bc_cbc_dx = self.gradients(bc_cbc, coords_bc)[:, 0]
        bc_cc_dx = self.gradients(bc_cc, coords_bc)[:, 0]
        loss_bc = torch.mean(bc_ca_dx**2) + torch.mean(bc_cbc_dx**2) + torch.mean(bc_cc_dx**2)

        '''Initial Condition Part'''
        coords_ic =coords_ic.clone().detach().requires_grad_(True)
        mask_h = (coords_ic[:, 0] <= self.h/self.L)
        tensor_h = coords_ic[mask_h].requires_grad_(True)

        mask_L = (coords_ic[:, 0] >= self.h/self.L) & (coords_ic[:, 0] <= self.L/self.L)      
        tensor_L = coords_ic[mask_L].requires_grad_(True)

        ca_in_h,cbc_in_h = self.dnn(tensor_h)[:,0],self.dnn(tensor_h)[:,1]
        ca_in_L,cbc_in_L = self.dnn(tensor_L)[:,0],self.dnn(tensor_L)[:,1]
        cc_init,cab_init,cabb_init,caab_init = self.dnn(coords_ic)[:,2],self.dnn(coords_ic)[:,3],self.dnn(coords_ic)[:,4],self.dnn(coords_ic)[:,5]

        loss_ic = torch.mean((ca_in_h - self.N_SA/self.N_SA)**2) + torch.mean((cbc_in_L - self.N_SB/self.N_SB)**2) +\
            torch.mean(cc_init **2) + torch.mean(cab_init **2) + torch.mean(cabb_init **2) + torch.mean(caab_init **2) +\
            torch.mean(ca_in_L**2) + torch.mean(cbc_in_h**2)
        
        loss_ic_group1 = torch.mean((ca_in_h - self.N_SA/self.N_SA)**2) + torch.mean((cbc_in_L - self.N_SB/self.N_SB)**2) + torch.mean(ca_in_L**2) + torch.mean(cbc_in_h**2)
        loss_ic_group2 = torch.mean(cc_init **2) + torch.mean(cab_init **2) + torch.mean(cabb_init **2) + torch.mean(caab_init **2)
        
        return loss_pde,loss_bc,loss_ic,D_star,loss_ic_group1,loss_ic_group2

    ##Normal Loss Function
    def loss_function(self,loss_pde,loss_bc,loss_ic):
        loss_total = loss_pde + loss_bc + loss_ic
        return loss_total
    
    ##Adaptive Weighted Loss Function
    def AW_loss_function(self,loss_pde,loss_bc,loss_ic1,loss_ic2):
        w_loss_ic1 = 1. / (self.var_ic1 ** 2 + 1. / self.lam_threshold) * loss_ic1
        w_loss_ic2 = 1. / (self.var_ic2 ** 2 + 1. / self.lam_threshold) * loss_ic2
        penalize_item = torch.log(self.var_ic1 ** 2 + 1. / self.lam_threshold) +\
                        torch.log(self.var_ic2 ** 2 + 1. / self.lam_threshold)
        loss_total = loss_pde + loss_bc +w_loss_ic1 + w_loss_ic2 + penalize_item
        return loss_total
    
    ##Usage of Sampler
    def fetch_sample(self, sampler, num_interior,num_boundary, num_initial,additional_points):
        coords_pde,coords_bc,coords_ic = sampler.generate_non_dimensional_data(num_interior,num_boundary, num_initial,additional_points)
        return  coords_pde,coords_bc,coords_ic

    ##1 time epoch training
    def epoch_train(self,num_interior,num_boundary, num_initial,additional_points):
        coords_pde,coords_bc,coords_ic = self.fetch_sample(self.sampler, num_interior,num_boundary, num_initial,additional_points)
        loss_pde,loss_bc,loss_ic,D_star,loss_ic_gr1,loss_ic_gr2 = self.diffusion_reaction(coords_pde,coords_bc,coords_ic)
        return loss_pde,loss_bc,loss_ic,D_star,loss_ic_gr1,loss_ic_gr2            

    ##Regard relative error as testing
    def relative_error(self,Nondimensional = True):
        x = np.linspace(0,1,100)
        t = np.linspace(0,1,100)
        ms_x,ms_t = np.meshgrid(x,t)

        x = np.ravel(ms_x).reshape(-1, 1)
        t = np.ravel(ms_t).reshape(-1, 1)
        pt_x = torch.from_numpy(x).float().requires_grad_(True).to(device)
        pt_t = torch.from_numpy(t).float().requires_grad_(True).to(device)
        X_star = torch.cat([pt_x,pt_t],1).to(device)
        X_star = X_star.clone().detach().requires_grad_(True)

        result = self.dnn(X_star).data.cpu().numpy()
        c_a,c_bc,c_c,c_ab,c_abb,c_aab = result[:,0],result[:,1],result[:,2],result[:,3],result[:,4],result[:,5]
        
        if Nondimensional==False:
            c_a,c_bc,c_c,c_ab,c_abb,c_aab = c_a*self.N_SA, c_bc*self.N_SB, c_c*self.N_SC, c_ab*self.N_SAB, c_abb*self.N_SABB, c_aab*0
            fdm_filenames = ['u1_output.csv', 'u2_output.csv', 'u3_output.csv', 'u4_output.csv', 'u5_output.csv','u6_output.csv']
            fdm_dir = "./Results/reaction t = 60/FDM_Reference_NonDimension"
            
        net_u1,net_u2,net_u3,net_u4,net_u5,net_u6 = [conc.reshape(100, 100) for conc in [c_a, c_bc, c_c, c_ab, c_abb, c_aab]]
        net_u1,net_u2,net_u3,net_u4,net_u5,net_u6 = [net_u.transpose() for net_u in [net_u1,net_u2,net_u3,net_u4,net_u5,net_u6]]
        """务必牢记这里对神经网络结果的转置操作"""

        fdm_filenames = ['u1_output.csv', 'u2_output.csv', 'u3_output.csv', 'u4_output.csv', 'u5_output.csv','u6_output.csv']

        fdm_dir = "./Results/reaction t = 60/FDM_Reference"

        fdm_u1,fdm_u2,fdm_u3,fdm_u4,fdm_u5,fdm_u6 = [pd.read_csv(f'{fdm_dir}/{filename}', encoding='utf-8', header=None) for filename in fdm_filenames]
        fdm_u1,fdm_u2,fdm_u3,fdm_u4,fdm_u5,fdm_u6 = [component.iloc[1:,1:] for component in [fdm_u1,fdm_u2,fdm_u3,fdm_u4,fdm_u5,fdm_u6]]

        error_u1 = np.linalg.norm(fdm_u1 - net_u1 , 2)/np.linalg.norm(fdm_u1, 2)
        error_u2 = np.linalg.norm(fdm_u2 - net_u2 , 2)/np.linalg.norm(fdm_u2, 2)
        error_u3 = np.linalg.norm(fdm_u3 - net_u3 , 2)/np.linalg.norm(fdm_u3, 2)
        error_u4 = np.linalg.norm(fdm_u4 - net_u4 , 2)/np.linalg.norm(fdm_u4, 2)
        error_u5 = np.linalg.norm(fdm_u5 - net_u5 , 2)/np.linalg.norm(fdm_u5, 2)
        # error_u6 = np.linalg.norm(fdm_u6 - net_u6 , 2)/np.linalg.norm(fdm_u6, 2)
        
        return error_u1,error_u2,error_u3,error_u4,error_u5
    
    '''Training Process'''    
    def train(self, nIter = 10000, num_interior = 100,num_boundary =250, num_initial = 500, additional_points = 0):
        maxIter = nIter

        start_time = timeit.default_timer()
        self.dnn.train()

        print(f"\n model: {self.model_type}, layer: {self.layers}")
        pbar = trange(nIter, ncols=240)    

        if self.model_type in ['PINN','IAW_PINN']:
            optimizer =  torch.optim.Adam(self.dnn.parameters(), lr = 1e-3,betas=(0.9, 0.999),eps=1e-8)  ##神经网络的优化器
            self.scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        if self.model_type in ['IA_PINN','I_PINN']:
            optimizer =  torch.optim.Adam(self.dnn.parameters(), lr = 1e-2,betas=(0.9, 0.999),eps=1e-8)
            self.scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        for it in pbar:
            current_lr = optimizer.param_groups[0]['lr']        
            
            def closuer():
                global loss_pde,loss_bc,loss_ic,D_star,loss_ic_gr1,loss_ic_gr2
                optimizer.zero_grad()
                if self.model_type in ['IAW_PINN', 'I_PINN']:
                    self.optimizer_ic.zero_grad()
            
                loss_pde,loss_bc,loss_ic,D_star,loss_ic_gr1,loss_ic_gr2 = self.epoch_train(num_interior,num_boundary, num_initial,additional_points)

                if self.model_type in ['PINN','IA_PINN']:
                    loss = self.loss_function(loss_pde,loss_bc,loss_ic)

                elif self.model_type in ['IAW_PINN', 'I_PINN']:
                    loss = self.AW_loss_function(loss_pde,loss_bc,loss_ic_gr1,loss_ic_gr2)

                loss.backward()

                ##Add gradient clipping, which is really heapful
                torch.nn.utils.clip_grad_norm_(self.dnn.parameters(), max_norm=1.0, norm_type= 2)  
                return loss                

            optimizer.step(closuer)

            error_u1,error_u2,error_u3,error_u4,error_u5 = self.relative_error()
            '''relative Error'''
            self.log_relative_u1.append(error_u1.item())
            self.log_relative_u2.append(error_u2.item())
            self.log_relative_u3.append(error_u3.item())
            self.log_relative_u4.append(error_u4.item())
            self.log_relative_u5.append(error_u5.item())

            if self.model_type in ['IAW_PINN', 'I_PINN']:
                self.optimizer_ic.step(closure=closuer)
            
            if self.model_type in ['PINN','IAW_PINN']:
                scheduler_decay = 500
            if self.model_type in ['IA_PINN','I_PINN']:
                scheduler_decay = 200
            
            if (self.iter>0) and (self.iter % scheduler_decay == 0):
                self.scheduler.step()
            
            loss_true = loss_pde + loss_bc +loss_ic
            ##If we meet gradient explosion, we stop the training
            if torch.isnan(loss_true):
                break

            self.loss_pde_log.append(loss_pde.item())
            self.loss_bc_log.append(loss_bc.item())
            self.loss_ic_log.append(loss_ic.item())
            self.loss_total_log.append(loss_true.item())            

            if it % 10 == 0:
                if self.model_type in ['PINN', 'IA_PINN']:
                    pbar.set_postfix({
                                      'Loss': '{0:.3e}'.format(loss_true.item()),
                                      'loss_pde': '{0:.3e}'.format(loss_pde.item()),
                                      'loss_bc': '{0:.3e}'.format(loss_bc.item()),
                                      'loss_ic': '{0:.3e}'.format(loss_ic.item()),
                                      'lr': '{0:.2e}'.format(current_lr),
                                      'D*': '{0:.2e},{1:.2e}'.format(torch.max(D_star),torch.min(D_star)),
                                      'grouped_ic':'{0:.2e},{1:.2e}'.format(loss_ic_gr1,loss_ic_gr2)                                         
                                      })
                    self.lam_ic1_log.append(1. / (self.var_ic1 ** 2).item())
                    self.lam_ic2_log.append(1. / (self.var_ic2 ** 2).item())
                    self.lam_ic_log.append(1. / (self.var_ic ** 2).item())
                
                elif self.model_type in ['IAW_PINN', 'I_PINN']:
                    pbar.set_postfix({
                                      'Loss': '{0:.3e}'.format(loss_true.item()),
                                      'loss_pde': '{0:.3e}'.format(loss_pde.item()),
                                      'loss_bc': '{0:.3e}'.format(loss_bc.item()),
                                      'loss_ic': '{0:.3e}'.format(loss_ic.item()),
                                      'lam_ic1': '{0:.2f}'.format(
                                          1. / (self.var_ic1 ** 2 + 1. / self.lam_threshold).item()),
                                      'lam_ic2': '{0:.2f}'.format(
                                          1. / (self.var_ic2 ** 2 + 1. / self.lam_threshold).item()),
                                      'lr': '{0:.2e}'.format(current_lr),
                                      'D*': '{0:.2e},{1:.2e}'.format(torch.max(D_star),torch.min(D_star)),
                                      'grouped_ic':'{0:.2e},{1:.2e}'.format(loss_ic_gr1,loss_ic_gr2)                                           
                                      })
                    self.lam_ic1_log.append(1. / (self.var_ic1 ** 2 + 1. / self.lam_threshold).item())
                    self.lam_ic2_log.append(1. / (self.var_ic2 ** 2 + 1. / self.lam_threshold).item())

            self.iter += 1 
        elapsed = timeit.default_timer() - start_time
        print("Time: {0:.2f}s".format(elapsed))        

    def predict_u(self, X_star):
        X_star = X_star.clone().detach().requires_grad_(True)
        u_pred = self.dnn(X_star)
        return u_pred                      

    def predict_r(self, X_star):
        X_star = torch.tensor(X_star, requires_grad=True).float().to(device)
        X_star = (X_star - self.mu_X) / self.sigma_X
        r_pred = self.net_r(X_star[:, 0:1], X_star[:, 1:2])
        return r_pred


