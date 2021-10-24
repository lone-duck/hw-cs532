import torch
import math
import operator as op

def pure_hashmap_update(d, k, v):
    if isinstance(k, torch.Tensor):
        k = k.item()
    d2 = d.copy()
    d2.update({k:v})
    return d2

# got a tip from Masoud on this
def vector_list_creation(args):
    try:
        return torch.stack(args)
    except:
        return args

def mat_transpose(args):
    if isinstance(args, tuple):
        if len(args) == 1:
            args = args[0]
        else:
            assert False, "Multi-element tuple???"
    try:
        return args.t()
    except:
        assert False, "whoops"

def mat_repmat(t, i1, i2):
    return t.repeat(i1.long().item(), i2.long().item())

# inspired by https://norvig.com/lispy.html
def eval_env():
    env = {}
    env.update({
        '+': torch.add,
        '-': torch.sub,
        '*': torch.mul,
        '/': torch.div,
        '>': torch.gt,
        '<': torch.lt,
        '>=': torch.ge,
        '<=': torch.le, 
        '=': torch.eq,
        'and': torch.logical_and,
        'or': torch.logical_or,
        'sqrt': torch.sqrt,
        'exp': torch.exp,
        'mat-tanh': torch.tanh,
        'mat-add': torch.add,
        'mat-mul': torch.matmul,
        'mat-repmat': mat_repmat,
        'mat-transpose': lambda *x: mat_transpose(x),
        'vector': lambda *x: vector_list_creation(x),
        'hash-map': lambda *x : dict(zip([i.item() if isinstance(i, torch.Tensor) else i for i in x[::2]], x[1::2])),
        'get': lambda x, y: x[y.long()] if isinstance(x, torch.Tensor) else x[y.item() if isinstance(y, torch.Tensor) else y],
        'put': lambda x, y, z: torch.cat((x[:y.long()], torch.tensor([z]), x[y.long()+1:])) if isinstance(x, torch.Tensor) else pure_hashmap_update(x,y,z),
        'append' : lambda x, y: torch.cat((x, torch.tensor([y]))),
        'first' : lambda x: x[0],
        'second': lambda x: x[1],
        'rest': lambda x: x[1:],
        'last' : lambda x: x[-1],
        'remove': lambda x, y : torch.cat((x[:y.long()], x[y.long()+1:])) if isinstance(x, torch.Tensor) else {i:x[i] for i in x if i != y},
        'normal': torch.distributions.Normal,
        'beta': torch.distributions.beta.Beta,
        'exponential': torch.distributions.exponential.Exponential,
        'uniform': torch.distributions.uniform.Uniform,
        'bernoulli': torch.distributions.bernoulli.Bernoulli,
        'flip': torch.distributions.bernoulli.Bernoulli,
        'discrete': lambda *x: torch.distributions.categorical.Categorical(x[0]),
        'dirichlet': lambda *x: torch.distributions.dirichlet.Dirichlet(x[0]), 
        'gamma': torch.distributions.gamma.Gamma 
        })


    return env
