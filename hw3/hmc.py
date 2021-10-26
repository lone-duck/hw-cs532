# hamiltonian_monte_carlo
# TODO: clean up ENV behaviour so that all this sample_from_graph, jll, deterministic_eval
# can be imported

import torch 
from primitives import eval_env
from graph_utils import topological_sort
import copy

ENV = eval_env()


def hmc(graph, S, X0=None, T=20, epsilon=0.1, m=1, return_samples=False, report_acceptance_rate=False):
    # NOTE: for now, assume that len(X) is dimensionality of R
    # this may not be the case if element of X are vectors themselves
    
    if return_samples:
        samples = [None]*S
    if report_acceptance_rate:
        accepts = 0
    
    jlls = [None]*S
    ret_vals = [None]*S

    if X0 is None:
        X0 = sample_from_prior(graph)

    Xprev = X0
    Xdim = len(X0)

    ret_expr = graph[2]

    R_dist = torch.distributions.MultivariateNormal(torch.zeros(Xdim), m*torch.eye(Xdim))
    unif = torch.distributions.uniform.Uniform(0, 1)

    for s in range(S):
        # get a new R
        R_vals = R_dist.sample()
        R = dict(zip(Xprev.keys(), R_vals))
        Xprime, Rprime = leapfrog(graph, copy.deepcopy(Xprev), copy.deepcopy(R), T, epsilon)
        # compute acceptance values
        u = unif.sample()
        alpha = accept(graph, Xprev, R, Xprime, Rprime, m)
        # based on acceptance values, choose which X to keep
        if u < alpha:
            if report_acceptance_rate:
                accepts +=1
            Xkeep = Xprime
        else:
            Xkeep = Xprev

        # store appropriate things
        if return_samples:
            samples[s] = Xkeep
        ret_vals[s] = deterministic_evaluate(ret_expr, Xkeep)[0]
        jlls[s] = joint_log_likelihood(Xkeep, graph)

        # update Xprev
        Xprev = Xkeep

    if report_acceptance_rate:
        print("Acceptance rate: {}".format(accepts/S))

    if return_samples:
        return ret_vals, jlls, samples
    else:
        return ret_vals, jlls


def leapfrog(graph, X, R, T, epsilon):
    gradU = get_grad_U(graph, X)
    R = R_update(R, gradU, epsilon, half_step=True)

    for t in range(T-1):
        X = X_update(X, R, epsilon)
        gradU = get_grad_U(graph, X)
        R = R_update(R, gradU, epsilon)

    X = X_update(X, R, epsilon)
    gradU = get_grad_U(graph, X)
    R = R_update(R, gradU, epsilon, half_step=True)
    return X, R


def get_grad_U(graph, X):

    P = graph[1]['P']
    Y = {k: torch.tensor(v).float() for k, v in graph[1]['Y'].items()}
    V = {**X, **Y}
    Vkeys = list(V.keys())
    U = 0
    for v in Vkeys:
        V[v].requires_grad=True
        if V[v].grad != None:
            V[v].grad.zero_()
    for v in Vkeys:
        expr = P[v][1]
        p_v, _ = deterministic_evaluate(expr, V)
        U += p_v.log_prob(V[v])

    U *= -1
    U.backward()
    gradU = {key: value.grad for key, value in X.items()}

    return gradU


def R_update(R, gradU, epsilon, half_step=False):
    
    keys = list(R.keys())
    if half_step:
        epsilon = 0.5*epsilon

    for r in keys:
        R[r] -= epsilon*gradU[r]

    return R


def X_update(X, R, epsilon):
    keys = list(X.keys())

    for x in keys:
        X[x].requires_grad = False
        X[x] += epsilon*R[x]

    return X


def accept(graph, X, R, Xprime, Rprime, m):
    
    K = 0
    for r in R.values():
        K += 0.5*r**2/m
    U = -1*joint_log_likelihood(X, graph)
    H = K + U

    Kprime = 0 
    for r in Rprime.values():
        Kprime += 0.5*r**2/m 
    Uprime = -1*joint_log_likelihood(Xprime, graph)
    Hprime = Kprime + Uprime

    return torch.exp(-Hprime + H)


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
    Y = {k: torch.tensor(v).float() for k, v in graph[1]['Y'].items()}
    V = {**X, **Y}
    Vkeys = list(V.keys())
    jll = 0

    for v in Vkeys:
        expr = P[v][1]
        p_v, _ = deterministic_evaluate(expr, V)
        jll += p_v.log_prob(V[v])

    return jll


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


