this script is using the gap statistics to run k-means algorithm for many times to 
find the best K value for the dataset.

because k-mean really depends on the initial points and thus the results can be different given different initial points; 
therefore use sklearn packages to run many times with different initial ponits, and this can be one parameter for the gap statistics.

this module should be imported into other python scripts and combined with sklearn to find the best K value.


parameters:

    refs: np.array or None, it is the replicated data that you want to compare with if there exists one; 
    if no existing replicated/proper data, just use None, and the function will automatically generates them; 
    
    B: int, the number of replicated samples to run gap-statistics; it is recommended as 10, and it should not be changed/decreased that to a smaller value;
    
    K: list, the range of K values to test on;
    
    N_init: int, states the number of initial starting points for each K-mean running under sklearn, in order to get stable clustering result each time; 
    you may not need such many starting points, so it can be reduced to a smaller number to quicken the computation;
    
    n_jobs: int, clarifies the parallel computing, could fasten the computation, this can be only changed inside the script, not as an argument of the function;


# to install
    pip install gapkmean


# to use as a module in python
    from gapkmean import gap

# to find the best K value of K-mean algorithm

    #note `data` should be an numpy.array
    gaps, s_k, K = gap.gap_statistic(data, refs=None, B=10, K=range(1,11), N_init = 10)
    bestKValue = gap.find_optimal_k(gaps, s_k, K)

    