import torch
import torch.distributions as dist
from pyrsistent import pmap,plist
from copy import deepcopy



class Normal(dist.Normal):
    
    def __init__(self, alpha, loc, scale):
        
        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)
        

def mat_repmat(t, i1, i2):
    return t.repeat(i1.long().item(), i2.long().item())

def push_addr(alpha, value):
    return alpha + value

def vector_list_creation(args):
    try:
        return torch.stack(args)
    except:
        return args

def pure_hashmap_update(d, k, v):
    if isinstance(k, torch.Tensor):
        k = k.item()
    d2 = deepcopy(d)
    d2.update({k:v})
    return d2


BASE_ENV = {
            'push-address' : push_addr,
            '+': torch.add,
            '-': torch.sub,
            '*': torch.mul,
            '/': torch.div,
            '>': torch.gt,
            '<': torch.lt,
            '>=': torch.ge,
            '<=': torch.le, 
            '=': torch.eq,
            'sqrt': torch.sqrt,
            'exp': torch.exp,
            'mat-tanh': torch.tanh,
            'mat-add': torch.add,
            'mat-mul': torch.matmul,
            'mat-repmat': mat_repmat,
            'vector' : lambda *x: vector_list_creation(x),
            'get': lambda x, y: deepcopy(x[y.long()]) if isinstance(x, torch.Tensor) else deepcopy(x[y.item() if isinstance(y, torch.Tensor) else y]),
            'put': lambda x, y, z: deepcopy(torch.cat((x[:y.long()], torch.tensor([z]), x[y.long()+1:]))) if isinstance(x, torch.Tensor) else pure_hashmap_update(x,y,z),
            'first' : lambda x: deepcopy(x[0]),
            'last' : lambda x: deepcopy(x[-1]),
            'append' : lambda x, y: deepcopy(torch.cat((x, torch.tensor([y])))),
            'hash-map': lambda *x : deepcopy(dict(zip([i.item() if isinstance(i, torch.Tensor) else i for i in x[::2]], x[1::2]))),
            }





