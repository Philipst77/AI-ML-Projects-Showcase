import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import time

# ———————————
# Load data
# ———————————
filePath1 = "test1_data.txt"
filePath2 = "test2_data.txt"
dataTesting1 = np.loadtxt(filePath1, delimiter=" ")
dataTesting2 = np.loadtxt(filePath2, delimiter=" ")

# ———————————
# Parameters
# ———————————
k = 2                       # number of clusters
var = 5                     # σ² in the RBF kernel
input_data = dataTesting1   # choose your dataset
initMethod = "byOriginDistance"  
# options: "random", "byCenterDistance", "byOriginDistance"


# ———————————
# Initialization
# ———————————
def initCluster(data, nCluster, method):
    clusters = [[] for _ in range(nCluster)]

    if method == "random":
        shuffled = data.copy()
        np.random.shuffle(shuffled)
        for i, x in enumerate(shuffled):
            clusters[i % nCluster].append(x)

    else:
        # pick reference point
        if method == "byCenterDistance":
            ref = np.mean(data, axis=0)
        else:
            ref = np.zeros(data.shape[1])
        # sort by distance to ref
        dists = np.linalg.norm(data - ref, axis=1)
        sorted_idx = np.argsort(dists)
        divider = data.shape[0] // nCluster

        for rank, idx in enumerate(sorted_idx):
            cluster_idx = min(rank // divider, nCluster - 1)
            clusters[cluster_idx].append(data[idx])

    return clusters


# ———————————
# Kernel & auxiliary terms
# ———————————
def RbfKernel(x, y, sigma):
    """ RBF kernel K(x,y) = exp(-||x - y||^2 / (2σ²)) """
    d = x - y
    sqdist = np.dot(d, d)
    return np.exp(-sqdist / (2 * sigma**2))


def thirdTerm(cluster_arr):
    """ (1 / n²) * sum_{i,j} K(x_i, x_j) """
    n = cluster_arr.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += RbfKernel(cluster_arr[i], cluster_arr[j], var)
    return total / (n * n)


def secondTerm(x_i, cluster_arr):
    """ (2 / n) * sum_j K(x_i, x_j) """
    n = cluster_arr.shape[0]
    s = sum(RbfKernel(x_i, cluster_arr[j], var) for j in range(n))
    return 2 * s / n


# ———————————
# Plotting
# ———————————
def plotResult(cluster_list, centroids, iteration, converged):
    plt.figure("k‐Means (kernel)")
    plt.clf()
    plt.title(f"Iteration {iteration}")
    colors = iter(cm.rainbow(np.linspace(0, 1, len(cluster_list))))

    # plot points
    for cidx, cluster in enumerate(cluster_list):
        col = next(colors)
        arr = np.vstack(cluster)
        plt.scatter(arr[:, 0], arr[:, 1], s=50, c=[col], marker='.')

    # plot centroids
    colors = iter(cm.rainbow(np.linspace(0, 1, len(cluster_list))))
    for cidx, center in enumerate(centroids):
        col = next(colors)
        plt.scatter(center[0], center[1], s=200, c=[col],
                    marker='*', edgecolors='k')

    if not converged:
        plt.pause(0.2)
    else:
        plt.show()


# ———————————
# Main k‐Means Kernel
# ———————————
def kMeansKernel(data, initMethod):
    iteration = 0
    clusters = initCluster(data, k, initMethod)

    while True:
        # 1) compute centroids (for visualization only)
        centroids = np.vstack([
            np.mean(np.vstack(cluster), axis=0)
            for cluster in clusters
        ])

        # 2) plot current state
        plotResult(clusters, centroids, iteration, converged=False)

        # 3) build distance‐like matrix via kernel
        N = data.shape[0]
        Kmat = np.empty((N, 0))
        for cluster in clusters:
            arr = np.vstack(cluster)
            t3 = thirdTerm(arr)
            col3 = np.full((N, 1), t3)
            col2 = np.array([secondTerm(data[i], arr) for i in range(N)])[:, None]
            # negative term2 + term3
            Kmat = np.hstack((Kmat, col3 - col2))

        # 4) reassign
        labels = np.argmin(Kmat, axis=1)
        new_clusters = [[] for _ in range(k)]
        for idx, lbl in enumerate(labels):
            new_clusters[int(lbl)].append(data[idx])

        # 5) check convergence
        converged = True
        for old, new in zip(clusters, new_clusters):
            if len(old) != len(new) or not np.allclose(np.vstack(old), np.vstack(new)):
                converged = False
                break

        if converged:
            # final plot
            plotResult(new_clusters, centroids, iteration, converged=True)
            return new_clusters, centroids, iteration

        # otherwise iterate
        clusters = new_clusters
        iteration += 1


if __name__ == "__main__":
    clusters, centroids, nIter = kMeansKernel(input_data, initMethod)
    print(f"Converged in {nIter} iterations")
