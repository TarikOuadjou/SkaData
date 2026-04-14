import numpy as np
from scipy.stats import qmc

prior_dic = {
    'ALPHA_STAR':       (-0.5, 1.0),
    'F_STAR10':         (-3.0, 0.0),
    'F_ESC10':          (-3.0, 1.0),
    'ALPHA_ESC':        (-1.0, 0.5),
    'M_TURN':           (8.0, 10.0),
    't_STAR':           (0.0,  1.0),
    'X_RAY_SPEC_INDEX': (0.5,  2.0),
}

N = 128  # must be a power of 2

keys = list(prior_dic.keys())
bounds = np.array([prior_dic[k] for k in keys])

sampler = qmc.Sobol(d=len(keys), scramble=True, seed=42)
unit_samples = sampler.random(n=N)
scaled = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])

samples = [dict(zip(keys, row)) for row in scaled]
np.save("sobol_samples.npy", samples)
print(f"Saved {N} Sobol samples to sobol_samples.npy")