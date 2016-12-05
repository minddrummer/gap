'''this script is using the gap statistics to run k-means algorithm for many times to 
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
'''
import numpy as np
import random
from sklearn.cluster import KMeans as Kmeans
import scipy
import sys

logging = False
def printLog(*log):
    '''
    printing function

    log: any type, is the non-key arguments
    '''
    if logging:
        print log

def init_board_gauss(N, k, clear = True):
    '''
    this function generates some random samples with k clusters, the return array has two features/cols

    N: int, the number of datapoints
    k: int, the number of clusters
    '''
    n = float(N)/k
    X = []
    if clear:
        for i in range(k):
            c = (random.uniform(-2, 2), random.uniform(-2, 2))
            s = random.uniform(0.05,0.25)
            x = []
            while len(x) < n:
                a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
                if abs(a) < 3 and abs(b) < 3:
                    x.append([a,b])
            X.extend(x)
    else:
        for i in range(k):
            c = (random.uniform(-1, 1), random.uniform(-1, 1))
            s = random.uniform(0.05,0.5)
            x = []
            while len(x) < n:
                a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
                if abs(a) < 1 and abs(b) < 1:
                    x.append([a,b])
            X.extend(x)
    X = np.array(X)[:N]
    return X


def short_pair_wise_D(each_cluster):
    '''
    this function computes the sum of the pairwise distance(repeatedly) of all points in one cluster;
    each pair be counting twice here; using the short formula below instead of the original meaning of pairwise distance of all points

    each_cluster: np.array, with containing all points' info within the array
    '''
    mu = each_cluster.mean(axis = 0)
    total = sum(sum((each_cluster - mu)**2)) * 2.0 * each_cluster.shape[0]
    return total

def compute_Wk(data, classfication_result):
    '''
    this function computes the Wk after each clustering

    data:np.array, containing all the data
    classfication_result: np.array, containing all the clustering results for all the data
    '''
    Wk = 0
    label_set = set(classfication_result.tolist())
    for label in label_set:
        each_cluster = data[classfication_result == label, :]
        D = short_pair_wise_D(each_cluster)
        Wk = Wk + D/(2.0*each_cluster.shape[0])
    return Wk
 
def gap_statistic(X, refs=None, B=10, K=range(1,11), N_init = 10):
    '''
    this function first generates B reference samples; for each sample, the sample size is the same as the original datasets;
    the value for each reference sample follows a uniform distribution for the range of each feature of the original datasets;
    using a simplify formula to compute the D of each cluster, and then the Wk; K should be a increment list, 1-10 is fair enough;
    the B value is about the number of replicated samples to run gap-statistics, it is recommended as 10, and it should not be changed/decreased that to a smaller value;
    
    X: np.array, the original data;
    refs: np.array or None, it is the replicated data that you want to compare with if there exists one; if no existing replicated/proper data, just use None, and the function
    will automatically generates them; 
    B: int, the number of replicated samples to run gap-statistics; it is recommended as 10, and it should not be changed/decreased that to a smaller value;
    K: list type, the range of K values to test on;
    N_init: int, states the number of initial starting points for each K-mean running under sklearn, in order to get stable clustering result each time; 
    you may not need such many starting points, so it can be reduced to a smaller number to fasten the computation;
    n_jobs below is not an argument for this function,but it clarifies the parallel computing, could fasten the computation, this can be only changed inside the script, not as an argument of the function;
    '''
    shape = X.shape
    if refs==None:
        tops = X.max(axis=0)
        bots = X.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops-bots))
        rands = scipy.random.random_sample(size=(shape[0],shape[1],B))
        for i in range(B):
            rands[:,:,i] = rands[:,:,i]*dists+bots
    else:
        rands = refs

    gaps = np.zeros(len(K))
    Wks = np.zeros(len(K))
    Wkbs = np.zeros((len(K),B))

    for indk, k in enumerate(K):
        # #setup the kmean clustering instance
        #n_jobs set up for parallel:1 mean No Para-computing; -1 mean all parallel computing
        #n_init is the number of times each Kmeans running to get stable clustering results under each K value
        k_means =  Kmeans(n_clusters=k, init='k-means++', n_init=N_init, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
        k_means.fit(X)
        classfication_result = k_means.labels_
        #compute the Wk for the classification result
        Wks[indk] = compute_Wk(X,classfication_result)
        
        # clustering on B reference datasets for each 'k' 
        for i in range(B):
            Xb = rands[:,:,i]
            k_means.fit(Xb)
            classfication_result_b = k_means.labels_
            Wkbs[indk,i] = compute_Wk(Xb,classfication_result_b)

    #compute gaps and sk
    gaps = (np.log(Wkbs)).mean(axis = 1) - np.log(Wks)        
    sd_ks = np.std(np.log(Wkbs), axis=1)
    sk = sd_ks*np.sqrt(1+1.0/B)
    return gaps, sk, K

def find_optimal_k(gaps, s_k, K):
    '''
    this function is finding the best K value given the computed results of gap-statistics

    gaps: np.array, containing all the gap-statistics results;
    s_k: float, the baseline value to minus with; say reference paper for detailed meaning;
    K: list, containing all the tested K values;
    '''
    gaps_thres = gaps - s_k
    #printLog(gaps_thres)
    below_or_above = (gaps[0:-1] >= gaps_thres[1:])
    printLog('here is the comparsion between gap_k and (gap_k+1 - s_k+1):', below_or_above)
    if below_or_above.any():
        optimal_k = K[below_or_above.argmax()]
    else:
        printLog('have NOT found the best k above the next k+1 threshold yet, use the last k instead')
        optimal_k = K[-1]
    return optimal_k

def main():
    pass

if __name__ == '__main__':
    # uncomment and test and compare the two gaps if you want
    # X = init_board_gauss(200,4, clear = True)
    # #X = init_board_gauss(200,4, clear = False)
    # plt.scatter(X[:,0],X[:,1])
    # gaps, sk, K = gap_statistic(X)    
    # printLog(find_optimal_k(gaps, sk, K))
    sys.exit(main())