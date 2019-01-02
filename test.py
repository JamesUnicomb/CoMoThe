import numpy as np
from CollectiveMotion import CollectiveMotion

cm = CollectiveMotion()


for J in np.arange(0.0, 0.2, 0.025):
    prob = cm.calculate_probability(J = J,
                                    N = 512,
                                    n_steps = 800)

    print J, np.sum(prob)
