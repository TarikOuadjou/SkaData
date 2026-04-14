from generate_data.model import model
import sys
import numpy as np
if __name__ == "__main__":
    job_id = int(sys.argv[1])  # passed by the job array

    samples = np.load("sobol_samples.npy", allow_pickle=True)
    theta = samples[job_id]

    print(f"[Job {job_id}] Running with theta: {theta}")
    ps2d_list, x_HI_mean_list = model(theta)

    np.save(f"results/job_{job_id}_ps2d.npy", ps2d_list)
    np.save(f"results/job_{job_id}_xHI.npy", x_HI_mean_list)
    print(f"[Job {job_id}] Done.")