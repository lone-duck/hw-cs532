# mh_within_gibbs_program_5
# metropolis hastings within gibbs

import torch 
from primitives import eval_env
from graph_utils import topological_sort

ENV = eval_env()

def sample_from_prior(graph):
    # get contents of graph
    fn_defs = graph[0]
    V = graph[1]['V']
    A = graph[1]['A']
    P = graph[1]['P']
    Y = graph[1]['Y']
    ret_vals = graph[2]
    
    # deal with fn_defs
    global ENV
    ENV = eval_env()
    for defn in fn_defs.items():
        f_name = defn[0]
        f_v_is = defn[1][1]
        f_expr = defn[1][2]
        ENV.update({f_name: (f_v_is, f_expr)})
    
    # get sorted V
    sorted_V = topological_sort(A, V)

    # compute each value in order
    l = {}
    for v in sorted_V:
        task, expr = P[v][0], P[v][1]
        if task == "sample*":
            dist, _ = deterministic_evaluate(expr, l)
            l.update({v: dist.sample()})

    return l


def joint_log_likelihood(X, graph):
    P = graph[1]['P']
    V = X
    Vkeys = list(V.keys())
    jll = 0

    for v in Vkeys:
        expr = P[v][1]
        p_v, _ = deterministic_evaluate(expr, V)
        jll += p_v.log_prob(V[v])

    return jll



def gibbs_sampling_p5(graph, S, X0=None, verbose=False, return_samples=False):

    if return_samples:
        samples = [None]*S
    
    jlls = [None]*S
    ret_vals = [None]*S

    # sample from proposal
    if X0 is None:
        X0 = sample_from_prior(graph)
        P = graph[1]['P']
        Xkeys = list(X0.keys())
        task, expr = P[Xkeys[0]][0], P[Xkeys[0]][1]
        assert task == "sample*", "Found observed variable in X???"
        q0, _ = deterministic_evaluate(expr, X0)
        X0[Xkeys[0]] = q0.sample()
        X0[Xkeys[1]] = 7 - X0[Xkeys[0]]

    Xprev = X0

    for s in range(S):
        if verbose:
            print(s)
        X, ret_vals[s] = block_gibbs_step(Xprev, graph, s)
        if return_samples:
            samples[s] = X
        jlls[s] = joint_log_likelihood(X, graph)
        Xprev = X

    if return_samples:
        return ret_vals, jlls, samples
    else:
        return ret_vals, jlls


def block_gibbs_step(X, graph, s):

    P = graph[1]['P']
    unif = torch.distributions.Uniform(0,1)
    Xkeys = list(X.keys())
    ret_vals = graph[2]

    if s % 2 == 0:
        sampled = 'sample1'
        deterministic = 'sample2'
    else:
        sampled = 'sample2'
        deterministic = 'sample1'

    # sample a new X from block proposal
    expr = P[sampled][1]
    q_sampled, _ =  deterministic_evaluate(expr, X)
    Xprime = X.copy()
    Xprime[sampled] = q_sampled.sample()
    Xprime[deterministic] = 7 - Xprime[sampled]
    # compute alpha
    alpha = accept(deterministic, Xprime, X, graph)
    u = unif.sample()
    if u < alpha:
        X = Xprime

    return X, deterministic_evaluate(ret_vals, X)[0]


def accept(x, Xprime, X, graph):

    P = graph[1]['P']
    log_alpha = 0.0

    expr = P[x][1]
    p, _ = deterministic_evaluate(expr, X)
    log_alpha += (p.log_prob(Xprime[x]) - p.log_prob(X[x]))

    return torch.exp(log_alpha)


def deterministic_evaluate(e, l, sig=None):
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
        exp = (conseq if deterministic_evaluate(test, l)[0] else alt)
        return deterministic_evaluate(exp, l)
    # let statements
    elif e[0] == 'let':
        # get symbol
        symbol = e[1][0]
        # get value of e1
        value, _ = deterministic_evaluate(e[1][1], l)
        # evaluate e2 with value 
        return deterministic_evaluate(e[2], {**l, symbol: value})
    # sample statement
    if e[0] == 'sample':
        dist = deterministic_evaluate(e[1], l)[0]
        # make sure it is a distribution object
        assert getattr(dist, '__module__', None).split('.')[:2] == ['torch', 'distributions']
        return dist.sample(), sig
    # obsere statements
    # TODO: change this, maybe in this hw or for hw3
    if e[0] == 'observe':
        dist = deterministic_evaluate(e[1], l)[0] # get dist
        y = deterministic_evaluate(e[2], l)[0]    # get observed value
        # make sure it is a distribution object
        assert getattr(dist, '__module__', None).split('.')[:2] == ['torch', 'distributions']
        # TODO: do something with observed value
        return dist.sample(), sig
    # procedure call, either primitive or user-defined
    else:
        result = deterministic_evaluate(e[0], l)
        proc, sig = result
        # primitives are functions
        if callable(proc):
            args = [deterministic_evaluate(arg, l)[0] for arg in e[1:]]
            result, sig = proc(*args), sig
            return result, sig
        # user defined functions are not
        else:
            # as written in algorithm 6
            v_is, e0 = proc 
            assert(len(v_is) == len(e[1:]))
            c_is = [deterministic_evaluate(arg, l)[0] for arg in e[1:]]
            l_proc = dict(zip(v_is, c_is))
            return deterministic_evaluate(e0, {**l, **l_proc})










