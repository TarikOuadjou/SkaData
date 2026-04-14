import numpy as np
import sys
import os
import traceback
from generate_data.model import model

def is_done(point_dir):
    """Check if this point was already computed successfully."""
    return os.path.exists(os.path.join(point_dir, "done.flag"))

def run_point(job_id):
    samples = np.load("sobol_samples.npy", allow_pickle=True)
    theta = samples[job_id]

    point_dir = f"results/point_{job_id:04d}"
    os.makedirs(point_dir, exist_ok=True)

    # Skip if already done (useful for resubmissions)
    if is_done(point_dir):
        print(f"[Job {job_id}] Already done, skipping.")
        return

    # Save theta immediately for traceability
    np.save(os.path.join(point_dir, "theta.npy"), theta)
    
    cache_dir = f"/gpfs/workdir/ouadjout/cache/job_{job_id:04d}"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[Job {job_id}] Running with theta: {theta}")
    try:
        ps2d_list, x_HI_mean_list = model(theta, cache_dir=cache_dir)

        np.save(os.path.join(point_dir, "ps2d.npy"), ps2d_list)
        np.save(os.path.join(point_dir, "xHI.npy"), x_HI_mean_list)

        # Mark as done only if everything succeeded
        open(os.path.join(point_dir, "done.flag"), "w").close()
        print(f"[Job {job_id}] Done successfully.")

    except Exception as e:
        # Save the error for debugging without crashing the whole array
        with open(os.path.join(point_dir, "error.log"), "w") as f:
            f.write(traceback.format_exc())
        print(f"[Job {job_id}] FAILED: {e}")
        sys.exit(1)  # SLURM marks this task as failed

if __name__ == "__main__":
    job_id = int(sys.argv[1])
    run_point(job_id)