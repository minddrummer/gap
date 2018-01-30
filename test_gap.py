from gap import gap
import numpy as np
from numpy.random import multivariate_normal

def test_gap():
    means = [np.array([1, 1]),
             np.array([5, 4]),
             np.array([-2, -3])]
    cov = np.identity(2)
    X = np.concatenate([multivariate_normal(m, cov, 100)
                        for m in means])
    assert X.shape == (300, 2)
    gaps, s_k, K = gap.gap_statistic(X,
				     refs=None,
                                     B=10,
                                     K=range(1,11),
                                     N_init=10)
    bestKValue = gap.find_optimal_k(gaps, s_k, K)
    assert bestKValue == 3
