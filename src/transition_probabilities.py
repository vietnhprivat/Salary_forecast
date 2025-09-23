import numpy as np

# These define the transition probabilities between each seniority
transition_matrices = {
    0: np.array([
        [0.00, 0.00, 1.00],   # from New_job
        [0.10, 0.10, 0.80],   # from Promotion
        [0.15, 0.15, 0.70],   # from Stay
    ]),
    1: np.array([
        [0.00, 0.00, 1.00],
        [0.075, 0.075, 0.85],
        [0.125, 0.125, 0.75],
    ]),
    2: np.array([
        [0.00, 0.00, 1.00],
        [0.05, 0.05, 0.90],
        [0.10, 0.10, 0.80],
    ]),
    3: np.array([
        [0.00, 0.00, 1.00],
        [0.00, 0.00, 1.00],
        [0.00, 0.00, 1.00],
    ]),
}

import numpy as np

transition_matrices2 = {
    0: np.array([
        [0.05, 0.05, 0.90],   # from New_job
        [0.10, 0.05, 0.85],   # from Promotion
        [0.10, 0.08, 0.82],   # from Stay
    ]),
    1: np.array([
        [0.05, 0.05, 0.90],
        [0.12, 0.05, 0.83],
        [0.10, 0.06, 0.84],
    ]),
    2: np.array([
        [0.05, 0.05, 0.90],
        [0.15, 0.05, 0.80],
        [0.08, 0.04, 0.88],
    ]),
    3: np.array([
        [0.00, 0.00, 1.00],
        [0.00, 0.00, 1.00],
        [0.00, 0.00, 1.00],
    ]),
}
