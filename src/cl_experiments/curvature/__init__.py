"""Pipeline step 2 -- curvature -> posterior width (sigma init).

``fisher`` estimates the diagonal Fisher (the local "admissible-zone" geometry);
``sigma_init`` maps it to per-weight posterior std for the BLR update.
"""

from cl_experiments.curvature.fisher import FisherMode, diagonal_fisher
from cl_experiments.curvature.sigma_init import laplace_sigma, sigma_from_curvature

__all__ = [
    "FisherMode",
    "diagonal_fisher",
    "laplace_sigma",
    "sigma_from_curvature",
]
