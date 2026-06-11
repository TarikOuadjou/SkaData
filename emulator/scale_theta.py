import numpy as np
PARAM_RANGES = [
    (-0.5,  1.0),   # ALPHA_STAR  — stellar mass – UV slope
    (-3.0,  0.0),   # F_STAR10   — log10 star formation efficiency
    (-3.0,  1.0),   # F_ESC10    — log10 escape fraction
    (-1.0,  0.5),   # ALPHA_ESC  — escape fraction slope
    ( 8.0, 10.0),   # M_TURN     — log10 turnover mass [M_sun]
    ( 0.0,  1.0),   # t_STAR     — star formation time-scale
]

def scale_thetas(thetas: np.ndarray) -> np.ndarray:
    """Scale theta from its original range to the range [0, 1].

    Args:
        thetas (np.ndarray): An array of shape (N, 6) containing the original theta values.

    Returns:
        np.ndarray: An array of shape (N, 6) containing the scaled theta values in the range [0, 1].
    """
    bounds = np.array(PARAM_RANGES)  # shape (6, 2)
    low, high = bounds[:, 0], bounds[:, 1]  # shape (6,) each
    return (thetas - low) / (high - low)


def unscale_thetas(scaled_thetas: np.ndarray) -> np.ndarray:
    """Unscale theta from the range [0, 1] back to its original range.

    Args:
        scaled_theta (np.ndarray): An array of shape (N, 6) containing the scaled theta values in the range [0, 1].

    Returns:
        np.ndarray: An array of shape (N, 6) containing the unscaled theta values in their original range.
    """
    bounds = np.array(PARAM_RANGES)  # shape (6, 2)
    low, high = bounds[:, 0], bounds[:, 1]  # shape (6,) each
    return scaled_thetas * (high - low) + low
