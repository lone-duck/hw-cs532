import json
from smc import SMC
import torch
import numpy as np
import pickle

i = 2
with open('asts/{}.json'.format(i),'r') as f:
    exp = json.load(f)

for power in range(6):
	n_particles = 10**power
	logZ, particles = SMC(n_particles, exp)
	values = torch.stack(particles).numpy()
	save_dict = {'logZ': logZ, 'values': values}
	fname = 'pickles/program{}/{}_particles.pkl'.format(str(i), str(n_particles))
	with open(fname, 'wb') as f:
		pickle.dump(save_dict, f)
	print("Done {} particle run".format(n_particles))