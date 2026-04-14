import numpy as np
from scipy.stats import qmc

prior_dic = {
    'ALPHA_STAR':        (-0.5, 1.0),
    'F_STAR10':          (-3.0, 0.0),
    'F_ESC10':           (-3.0, 1.0),
    'ALPHA_ESC':         (-1.0, 0.5),
    'M_TURN':            (8.0, 10.0),
    't_STAR':            (0.0,  1.0),
    'X_RAY_SPEC_INDEX':  (0.5,  2.0),
}

def generate_sobol_samples(prior_dic, n_samples=4, seed=42):
    keys = list(prior_dic.keys())
    bounds = np.array([prior_dic[k] for k in keys])  # (n_params, 2)

    sampler = qmc.Sobol(d=len(keys), scramble=True, seed=seed)
    # n_samples must be a power of 2 for Sobol — 4 is fine
    unit_samples = sampler.random(n=n_samples)           # (n_samples, n_params) in [0, 1]
    scaled = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])  # rescale to prior bounds

    samples = [dict(zip(keys, row)) for row in scaled]
    return samples

if __name__ == "__main__":
    samples = generate_sobol_samples(prior_dic, n_samples=4)
    for i, s in enumerate(samples):
        print(f"Sample {i}: {s}")

    # Save for the job array
    np.save("sobol_samples.npy", samples)
    print("Saved to sobol_samples.npy")
    