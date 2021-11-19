from evaluator import evaluate
import torch
import numpy as np
import json
import sys

def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res


def resample_particles(particles, log_weights):
    log_weights = torch.stack(log_weights)
    weights = torch.exp(log_weights)
    L = len(log_weights)
    cat = torch.distributions.Categorical(probs=weights/torch.sum(weights))
    new_particles = [None]*L 
    indices = cat.sample_n(L)
    for i, index in enumerate(indices):
        new_particles[i] = particles[index.item()]
    logZ = torch.log(torch.mean(weights))
    return logZ, new_particles


def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.
        particles.append(res)
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    while not done:
        print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                #observe case
                assert 'done' not in res[2], "done found in res[2]"
                cont, args, sigma = res
                # check address
                if i == 0:
                    alpha = sigma['alpha']
                else:
                    assert alpha == sigma['alpha'], "Found particle at different address"
                # update particles
                particles[i] = res 
                # compute weights
                d = sigma['d']
                c = sigma['c']
                weights[i] = d.log_prob(c)
                
        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


if __name__ == '__main__':

    for i in range(1,5):
        with open('asts/{}.json'.format(i),'r') as f:
            exp = json.load(f)
        n_particles = None #TODO 
        logZ, particles = SMC(n_particles, exp)

        print('logZ: ', logZ)

        values = torch.stack(particles)
        #TODO: some presentation of the results
