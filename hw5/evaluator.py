from primitives import BASE_ENV
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap,plist
import copy
import json
import torch


def evaluate_outer(e, env=None): #TODO: add sigma, or something
    
    # create env if it doesn't exist
    if env is None:
        env = standard_env()

    # evaluate, which turns into a single fn
    proc = evaluate(e, env)

    # call fn with alpha = ''
    return proc('')

def evaluate(e, env):
    #evaluate expression
    #base cases
    if isinstance(e, (int, float)):
        return torch.tensor(float(e))
    elif isinstance(e, str):
        try:
            return env.find(e)[e]
        except:
            assert False, "Could not find " + e
    # is there another case?
    else:
        assert isinstance(e, list), "Found unexpected base case: {}".format(e)

    # recursive cases
    op, *args = e
    if op == 'fn':
        params = args[0]
        body = args[1]
        return Procedure(params, body, copy.deepcopy(env))
    else:
        proc = evaluate(op, env)
        addr = args[0]
        args = args[1:]
        vals = [evaluate(arg, env) for arg in args]
        return proc(*vals)


# copied from Lispy
class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, params=(), args=(), outer=None):
        self.update(zip(params, args))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, params, body, env):
        self.params, self.body, self.env = params, body, env
    def __call__(self, *args): 
        # todo: make deep copies?
        return evaluate(self.body, Env(self.params, args, self.env))


def standard_env():
    
    env = Env()
    env.update(BASE_ENV)
    env.update({'alpha' : ''}) 

    return env


def get_stream(exp):
    while True:
        yield evaluate_outer(exp)


def run_deterministic_tests():
    
    for i in range(1,14):

        print("Starting FOPPL test {}".format(i))

        with open("programs/tests/deterministic/test_{}.json".format(str(i)), 'rb') as f:
            exp = json.load(f)
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate_outer(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print("FOPPL test {} passed".format(i))    
        
    print('FOPPL Tests passed')
        
    for i in range(1,13):

        with open("programs/tests/hoppl-deterministic/test_{}.json".format(str(i)), 'rb') as f:
            exp = json.load(f)
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate_outer(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        with open("programs/tests/probabilistic/test_{}.json".format(str(i)), 'rb') as f:
            exp = json.load(f)
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    run_deterministic_tests()
    run_probabilistic_tests()
    

    for i in range(1,4):
        print(i)
        with open("programs/{}.json".format(str(i)), 'rb') as f:
            exp = json.load(f)
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_outer(exp))        
