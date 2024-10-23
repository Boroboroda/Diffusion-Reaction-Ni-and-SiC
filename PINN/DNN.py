'''
@Project ：Diffusion Reaction PDE
@File    ：DNN.py
@IDE     ：VS Code
@Author  ：Cheng
@Date    ：2024-10-21
'''

import torch
import torch.optim
import torch.nn as nn
import numpy as np
import random

##Set Random Seeds
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(123)  ##I tried to set the seed to 123 and 42

'''Oringinal PINN:
   Although SiLu() will get a better result, but it is not stable in my case.
   In order to stabilize the training process, tanh() is selected as the activation function for the original PINN.
'''
class PINN(nn.Module):
    def __init__(self, layers, activation=torch.nn.functional.tanh,use_layernorm=False):
        super(PINN, self).__init__()
        #Size of Network
        self.layers = layers
        self.activation = activation
        self.use_layernorm = use_layernorm

        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        #Xavier-Initialization, with Normal-Distribution
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)
        
        # If LayerNorm is used, then use it
        if self.use_layernorm:
            self.layernorms = nn.ModuleList([nn.LayerNorm(layers[i + 1]) for i in range(len(layers) - 2)])

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)  # Ensure input is float32  
        a = self.activation(self.linear[0](x))
        
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            
            # If LayerNorm is used, then use it
            if self.use_layernorm:
                z = self.layernorms[i - 1](z)            
            a = self.activation(z)
        
        # Last-layer to constrain the output to be positive
        a = self.linear[-1](a)
        output = torch.nn.functional.softplus(a)  
        return output

'''Attention PINN:
   For this part, Activition function is SiLu(). Which can get a better result than tanh().
   And we introduce two encoders, attention_1 and attention_2, which are added behind each layer.
   With these, to realize the attention mechanism, a kind of weighted average,to better capture information.
'''

class Attention_PINN(nn.Module):
    def __init__(self, layers,activation=torch.nn.functional.silu):
        super(Attention_PINN, self).__init__()
        self.layers = layers
        self.activation = activation
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])


        """Attention Layer"""
        self.attention1 = nn.Linear(layers[0], layers[1])
        self.attention2 = nn.Linear(layers[0], layers[1])        

        for i in range(len(layers) - 1):
            nn.init.kaiming_normal_(self.linear[i].weight.data, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.linear[i].bias.data)
        
        nn.init.kaiming_normal_(self.attention1.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.attention1.bias.data)
        nn.init.kaiming_normal_(self.attention2.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.attention2.bias.data)
    
    """Here we introduce two encoder, to capture the feature of the input."""
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))

        encoder_1 = self.activation(self.attention1(x))
        encoder_2 = self.activation(self.attention2(x))
        
        a = a * encoder_1 + (1 - a) * encoder_2
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
            a = a * encoder_1 + (1 - a) * encoder_2
        a = self.linear[-1](a)
        output = torch.nn.functional.softplus(a)
        return output
