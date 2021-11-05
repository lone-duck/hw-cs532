# black box variation inference

import torch 
from primitives import eval_env
from graph_utils import topological_sort
import datetime
import wandb
import time
import numpy as np

ENV = eval_env()

def init_Q(graph):
    """
    Initialize proposal distributions for bbvi
    Args:
        graph: graph dictionary
    Output:
        a dictionary Q containing initial proposal distributions
    """

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

    Q = {}

    # find each q in order
    l = {}
    for v in sorted_V:
        task, expr = P[v][0], P[v][1]
        if task == "sample*":
            dist, _ = deterministic_evaluate(expr, l)
            l.update({v: dist.sample()})
            q = make_q(dist)
            Q.update({v: q})

    return Q

def make_q(d):
    
    # create a new q:
    if type(d) == torch.distributions.bernoulli.Bernoulli:
        probs = d.probs.clone().detach()
        q = distributions.Bernoulli(probs = probs)
    elif type(d) == torch.distributions.categorical.Categorical:
        probs = d.probs.clone().detach()
        q = distributions.Categorical(probs=probs)
    elif type(d) == torch.distributions.normal.Normal:
        loc = d.loc.clone().detach() 
        scale = d.scale.clone().detach()
        q = distributions.Normal(loc=loc, scale=scale)
    elif type(d) == torch.distributions.gamma.Gamma:
        concentration = d.concentration.clone().detach()
        rate = d.rate.clone().detach()
        q = distributions.Gamma(concentration=concentration, rate=rate)
    elif type(d) == torch.distributions.dirichlet.Dirichlet:
        concentration = d.concentration.clone().detach()
        q = distributions.Dirichlet(concentration=concentration)
    else:
        assert False, "Unknown distribution type: {}".format(type(d))

    return q


def bbvi_train(graph, T, L, base_string, Q=None, time_based=False, time_T=3600, lr=0.1, no_b=False, logging=True):
    """
    Trains BBVI proposal distributions.
    Args:
        graph: the graph denoting the problem
        T: number of outer training loops
        L: number of samples to use in gradient estimate
        Q: proposal distributions to start from... allows for 
           continuation of training from previously trained Q
    Returns:
        a new dictionary Q containing learned proposals
    """

    best_elbo = -np.inf
    
    if time_based:
        start = time.time()
    project_name = base_string
    if logging:
        wandb.init(project=project_name, entity="lone-duck")

    if Q is None:
        Q = init_Q(graph)

    # set up ENV
    fn_defs = graph[0]
    global ENV
    ENV = eval_env()
    for defn in fn_defs.items():
        f_name = defn[0]
        f_v_is = defn[1][1]
        f_expr = defn[1][2]
        ENV.update({f_name: (f_v_is, f_expr)})

    # get P, Y, sorted V
    P = graph[1]['P']
    V = graph[1]['V']
    Y = {k: torch.tensor(v).float() for k, v in graph[1]['Y'].items()}
    Xkeys = list(set(V) - set(Y.keys()))
    sorted_V = topological_sort(graph[1]['A'], V)

    # for t iterations, or for time_T seconds
    for t in range(T):
        # initiliaze lists for logW, G
        logWs = [None]*L
        Gs = [None]*L
        # "evaluate", i.e. sample from proposal and get logWs, Gs
        for l in range(L):
            logWs[l], Gs[l] = evaluation(P, Q, Y, sorted_V)
        # compute noisy elbo gradients
        g = elbo_gradients(logWs, Gs, L, Xkeys, no_b)
        # compute elbo
        elbo = compute_elbo(logWs)
        if elbo > best_elbo:
            best_Q = Q
            best_elbo = elbo
            print("new best elbo:")
            print(elbo)
        # do an update
        Q = update_Q(Q, g, t+1, lr)
        if logging:
            wandb.log({"ELBO": elbo})
        if time_based:
            if time.time() - start > time_T:
                break

    return best_Q


def update_Q(Q, g, t, lr):

    alpha = lr/torch.sqrt(torch.tensor(1.0*t))
    new_Q = {}

    for v in Q.keys():
        old_params = Q[v].Parameters()
        gradient = g[v]
        new_params = [(p + alpha*grad).clone().detach() for p, grad in zip(old_params, gradient)]
        new_Q[v] = type(Q[v])(*new_params, copy=True)

    return new_Q

    
def compute_elbo(logWs):
    
    return torch.mean(torch.stack(logWs))


def elbo_gradients(logWs, Gs, L, Xkeys, no_b):

    g = {}

    for v in Xkeys:
        # compute Fv's, Gv's for this v
        Fv = [None]*L
        Gv = [None]*L
        for l in range(L):
            if v in Gs[l]:
                Fv[l] = torch.stack(Gs[l][v]) * logWs[l]
                Gv[l] = torch.stack(Gs[l][v])
            else:
                Fv[l], Gv[l] = torch.tensor(0.), torch.tensor(0.)
        # both of shape (L, G.size())
        Fv = torch.stack(Fv)
        Gv = torch.stack(Gv)
        assert Fv.dim() < 3, "Need to ensure things work for higher dimensions"
        assert Gv.dim() < 3, "Need to ensure things work for higher dimensions"
        # compute b for this v
        if no_b:
            b = 0
        else:
            b = compute_b(Fv, Gv)
        g[v] = torch.sum(Fv - b*Gv, dim=0)/L

    return g

def compute_b(F, G):

    L, d = F.size()

    if d == 1:
        num = torch.sum((F - torch.mean(F))*(G - torch.mean(G)))/(L-1)
        den = torch.std(G)**2
    else:
        num = torch.tensor(0.)
        den = torch.tensor(0.)
        for i in range(d):
            num += torch.sum((F[:,i] - torch.mean(F[:,i]))*(G[:,i] - torch.mean(G[:,i])))/(L-1)
            den += torch.std(G[:,i])**2

    return num/den 


def evaluation(P, Q, Y, sorted_V):
    logW = 0
    G = {}
    l = {}

    for v in sorted_V:
        task, expr = P[v][0], P[v][1]
        if task == "sample*":
            # get prior dist
            d, _ = deterministic_evaluate(expr, l)
            # get proposal and grad-able proposal
            q = Q[v]
            q_with_grad = q.make_copy_with_grads()
            # take sample from proposal and add to l
            c = q.sample()
            l.update({v: c})
            # update logW
            with torch.no_grad():
                logW += d.log_prob(c) - q.log_prob(c)
            # get gradient
            log_prob_q = q_with_grad.log_prob(c)
            log_prob_q.backward()
            G[v] = [param.grad for param in q_with_grad.Parameters()]
        elif task == "observe*":
            # get prior dist, add log prob of observation to logW
            d, _ = deterministic_evaluate(expr, l)
            c = Y[v]
            with torch.no_grad():
                logW += d.log_prob(c)

    return logW, G


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










