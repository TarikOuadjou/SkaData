theta_fid1 = {
    'ALPHA_STAR': 0.5,
    'F_STAR10':  -1.3,
    'F_ESC10':   -1.0,
    'ALPHA_ESC': -0.5,
    'M_TURN':     8.7,   
    't_STAR':     0.5,
}


import numpy as np
import sys
import os
import traceback


def is_done(point_dir):
    """Check if this point was already computed successfully."""
    return os.path.exists(os.path.join(point_dir, "done.flag"))

def run_point(job_id):
    theta = theta_fid1

    point_dir = f"gaussian_test1/point_{job_id:04d}"
    os.makedirs(point_dir, exist_ok=True)

    # Skip if already done (useful for resubmissions)
    if is_done(point_dir):
        print(f"[Job {job_id}] Already done, skipping.")
        return
    from generate_data.model import model
    # Save theta immediately for traceability
    np.save(os.path.join(point_dir, "theta.npy"), theta)
    
    cache_dir = f"/gpfs/workdir/ouadjout/cache/job_{job_id:04d}"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[Job {job_id}] Running with theta: {theta}")
    try:

        np.random.seed(1234+job_id)  # Ensure reproducibility for this job
        random_number = np.random.randint(0, 2**31)
        np.save(os.path.join(point_dir, "random.npy"), random_number)
        ps2d_list, x_HI_mean_list = model(theta, cache_dir=cache_dir,seed=random_number)

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