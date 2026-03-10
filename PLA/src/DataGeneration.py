
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# generate 2D data which is lineraly separable
X, y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=1.8)

# convert labels to -1 and +1
y = np.where(y == 0, -1, +1)

# plot
plt.figure(figsize=(8,8))
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='C0', label='class -1', alpha=0.7)
plt.scatter(X[y == +1, 0], X[y == +1, 1], c='C1', label='class +1', alpha=0.7)
plt.legend()
plt.title('2D random data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.grid(True)
plt.show() 