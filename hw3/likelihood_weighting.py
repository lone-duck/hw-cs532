# evaluation based likelihood weighting

from daphne import daphne
from primitives import eval_env
import torch
import pickle

ENV = None

def likelihood_weighting_with_report(ast, L):
    result = likelihood_weighting(ast, L)
    returns = torch.stack(result['returns']).float()
    log_weights = torch.stack(result['log_weights'])
    M = torch.max(log_weights)
    normalized_weights = torch.exp(log_weights - M)/torch.sum(torch.exp(log_weights - M))
    
    if returns.dim() == 2:
        weighted_returns = normalized_weights.unsqueeze(dim=1)*returns
        expectation = torch.sum(weighted_returns, dim=0)
        var00 = torch.sum(normalized_weights*returns[:,0]**2) - expectation[0]**2
        var00 = torch.sum(normalized_weights*returns[:,1]**2) - expectation[1]**2
        var01 = torch.sum(normalized_weights*returns[:,0]*returns[:,1]) - expectation[0]*expectation[1]
        variance = torch.tensor([[var00, var01],[var01, var11]])
    else:
        weighted_returns = normalized_weights*returns
        expectation = torch.sum(weighted_returns)
        variance = torch.sum(normalized_weights*returns**2) - expectation**2

    
    print(expectation)
    print(variance)

def likelihood_weighting_and_save(ast, L, filename):
    result = likelihood_weighting(ast, L)
    f = open(filename, 'wb')
    pickle.dump(result, f)
    f.close()


def likelihood_weighting(ast, L):
    """
    Generate likelihood weighted samples from a program desugared by Daphne.
    Args:
        ast: json FOPPL program
        L: number of samples to generate
    Return:
        L samples and likelihood weights in a dictionary
    """
    returns = [None]*L
    log_weights = [None]*L

    for l in range(L):
        returns[l], log_weights[l] = evaluate_program(ast)

    return {'returns': returns, 'log_weights': log_weights}


def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: 
        samples with likelihood weights
    """
    global ENV
    ENV = eval_env()
    for defn in ast[:-1]:
        f_name = defn[1]
        f_v_is = defn[2]
        f_expr = defn[3]
        ENV.update({f_name: (f_v_is, f_expr)})
    l = {}
    sig = {'logW': 0}
    ret, sig = evaluate(ast[-1], l, sig)
    return ret, sig['logW']

# inspired by https://norvig.com/lispy.html
def evaluate(e, l, sig):
    # variable reference OR procedure OR just a string
    if isinstance(e, str):        
        # global procedures take precedence over locally defined vars
        if e in ENV:
            return ENV[e], sig
        elif e in l:
            return l[e], sig
        # could allow for hashmaps with string keys; for debugging setting this to fail
        else:
            assert False, "Unknown symbol: {}".format(e)
    # constant number
    elif isinstance(e, (int, float)):   
        return torch.tensor(float(e)), sig
    # if statements
    elif e[0] == 'if':
        (_, test, conseq, alt) = e
        test_value, sig = evaluate(test, l, sig)
        expr = (conseq if test_value else alt)
        return evaluate(expr, l, sig)
    # let statements
    elif e[0] == 'let':
        # get symbol
        symbol = e[1][0]
        # get value of e1
        value, sig = evaluate(e[1][1], l, sig)
        # evaluate e2 with value 
        return evaluate(e[2], {**l, symbol: value}, sig)
    # sample statement
    if e[0] == 'sample':
        dist, sig = evaluate(e[1], l, sig)
        # make sure it is a distribution object
        assert getattr(dist, '__module__', None).split('.')[:2] == ['torch', 'distributions']
        return dist.sample(), sig
    # observe statements
    if e[0] == 'observe':
        dist, sig = evaluate(e[1], l, sig) # get dist
        y, sig = evaluate(e[2], l, sig)    # get observed value
        # make sure it is a distribution object
        assert getattr(dist, '__module__', None).split('.')[:2] == ['torch', 'distributions']
        sig['logW'] = sig['logW'] + dist.log_prob(y)
        return y, sig
    # procedure call, either primitive or user-defined
    else:            
        proc, sig = evaluate(e[0], l, sig)
        # primitives are functions
        if callable(proc):
            args = [None]*len(e[1:])
            for i, arg in enumerate(e[1:]):
                result, sig = evaluate(arg, l, sig)
                args[i] = result
            result = proc(*args)
            return result, sig
        # user defined functions are not
        else:
            # as written in algorithm 6
            v_is, e0 = proc 
            assert(len(v_is) == len(e[1:]))
            c_is = [None]*len(e[1:])
            for i, arg in enumerate(e[1:]):
                result, sig = evaluate(arg, l, sig)
                c_is[i] = result
            l_proc = dict(zip(v_is, c_is))
            return evaluate(e0, {**l, **l_proc}, sig)