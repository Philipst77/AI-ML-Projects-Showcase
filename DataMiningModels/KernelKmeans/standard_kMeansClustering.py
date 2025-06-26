import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import time

# ———————————
# Load data
# ———————————
filePath1 = "test1_data.txt"
filePath2 = "test2_data.txt"
dataTesting1 = np.loadtxt(filePath1, delimiter=" ")
dataTesting2 = np.loadtxt(filePath2, delimiter=" ")
print("data testing: ", dataTesting2.shape)

# ———————————
# Parameters
# ———————————
k = 2                              # number of clusters
input_data = dataTesting1         # switch to dataTesting2 if you like
initCentroidMethod = "badInit"    # options: random, kmeans++, badInit, zeroInit


# ———————————
# Centroid Initialization
# ———————————
def initCentroid(data, method, k):
    n, d = data.shape
    if method == "random":
        # pick k random points
        idx = np.random.choice(n, k, replace=False)
        centroids = data[idx]

    elif method == "kmeans++":
        centroids = np.empty((0, d))
        # 1) pick one at random
        first = np.random.randint(n)
        centroids = np.vstack([centroids, data[first]])
        # 2) pick farthest-from-existing repeatedly
        for _ in range(1, k):
            # compute squared distances to nearest centroid
            dist_sq = np.min([np.sum((data - c) ** 2, axis=1) for c in centroids], axis=0)
            next_idx = int(np.argmax(dist_sq))
            centroids = np.vstack([centroids, data[next_idx]])

    elif method == "badInit":
        centroids = np.empty((0, d))
        data_copy = data.copy()
        # 1) pick one at random, remove from pool
        first = np.random.randint(data_copy.shape[0])
        centroids = np.vstack([centroids, data_copy[first]])
        data_copy = np.delete(data_copy, first, axis=0)
        # 2) pick nearest-to-existing clusters
        for _ in range(1, k):
            dist = np.min([np.linalg.norm(data_copy - c, axis=1) for c in centroids], axis=0)
            next_idx = int(np.argmin(dist))
            centroids = np.vstack([centroids, data_copy[next_idx]])
            data_copy = np.delete(data_copy, next_idx, axis=0)

    elif method == "zeroInit":
        centroids = np.zeros((k, d))

    else:
        raise ValueError(f"Unknown init method '{method}'")

    # plot initial centroids
    plt.figure("centroid initialization")
    plt.title(f"Init: {method}")
    plt.scatter(data[:, 0], data[:, 1], marker=".", s=100, alpha=0.6)
    cols = iter(cm.rainbow(np.linspace(0, 1, k)))
    for i, c in enumerate(centroids):
        col = next(cols)
        plt.scatter(c[0], c[1], marker="*", s=300, c=[col], edgecolors="k")
        plt.text(c[0], c[1], str(i+1), fontsize=16, weight="bold")
    plt.show()

    return centroids


# ———————————
# Plotting helper
# ———————————
def plotClusterResult(clusters, centroids, iteration, converged=False):
    plt.figure("kMeans ― iteration " + str(iteration))
    plt.clf()
    cols = iter(cm.rainbow(np.linspace(0, 1, len(clusters))))

    # scatter each cluster
    for i, cluster in enumerate(clusters):
        col = next(cols)
        pts = np.vstack(cluster)
        plt.scatter(pts[:, 0], pts[:, 1], marker=".", s=80, c=[col], alpha=0.6)

    # scatter centroids
    cols = iter(cm.rainbow(np.linspace(0, 1, len(clusters))))
    for i, c in enumerate(centroids):
        col = next(cols)
        plt.scatter(c[0], c[1], marker="*", s=300, c=[col], edgecolors="k")
    plt.title(f"Iteration {iteration}" + (" (converged)" if converged else ""))
    if converged:
        plt.show()
    else:
        plt.pause(0.2)


# ———————————
# Standard k-Means
# ———————————
def kMeans(data, centroids_init):
    centroids = centroids_init.copy()
    iteration = 0

    while True:
        iteration += 1
        # 1) assign each point to nearest centroid
        dists = np.stack([np.linalg.norm(data - c, axis=1) for c in centroids], axis=1)
        labels = np.argmin(dists, axis=1)

        clusters = [[] for _ in range(len(centroids))]
        for idx, lbl in enumerate(labels):
            clusters[int(lbl)].append(data[idx])

        # 2) recompute centroids
        new_centroids = np.vstack([
            np.mean(np.vstack(cluster), axis=0) if cluster else centroids[i]
            for i, cluster in enumerate(clusters)
        ])

        # 3) plot intermediate result
        plotClusterResult(clusters, new_centroids, iteration, converged=False)
        time.sleep(0.5)

        # 4) check convergence
        if np.allclose(new_centroids, centroids):
            plotClusterResult(clusters, new_centroids, iteration, converged=True)
            return clusters, new_centroids, iteration

        centroids = new_centroids


# ———————————
# Run
# ———————————
if __name__ == "__main__":
    centroidInit = initCentroid(input_data, initCentroidMethod, k)
    clusters, centroidFinal, nIter = kMeans(input_data, centroidInit)
    print(f"Converged in {nIter} iterations!")
