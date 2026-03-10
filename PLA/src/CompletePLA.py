import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def pla_train(X, y, max_epochs=1000, alpha=0.1, shuffle=True, random_state=None):
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape

    #vector X
    x0 = np.ones(n_samples)  
    x1 = X[:, 0]              
    x2 = X[:, 1]             

    # Stack into augmented matrix: each row is [x0, x1, x2]
    X_aug = np.column_stack((x0, x1, x2))

    w = np.array([0.0, 0.0, 0.0])  # initialize weight vector [w0, w1, w2]

    n_updates = 0
    for epoch in range(1, max_epochs + 1):
        errors = 0
        indices = np.arange(n_samples)
        if shuffle:
            rng.shuffle(indices)

        for i in indices:
            xi = X_aug[i]          # [x0_i, x1_i, x2_i]
            yi = y[i]
            pred = np.sign(w.dot(xi))
            # treat sign(0) as misclassification (so we update if pred == 0)
            if pred != yi:
                w += alpha * yi * xi
                n_updates += 1
                errors += 1

        # Stop if linearly separable on training set
        if errors == 0:
            return w, n_updates, epoch

    return w, n_updates, max_epochs

def plot_data_and_boundary(X, y, w, title="PLA result"):
    plt.figure(figsize=(8,8))
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='C0', label='class -1', alpha=0.7)
    plt.scatter(X[y == +1, 0], X[y == +1, 1], c='C1', label='class +1', alpha=0.7)

    # Decision boundary: w0 + w1*x + w2*y = 0  => y = -(w0 + w1*x)/w2
    if abs(w[2]) > 1e-8:
        x_vals = np.linspace(X[:,0].min() - 1, X[:,0].max() + 1, 200)
        y_vals = -(w[0] + w[1] * x_vals) / w[2]
        plt.plot(x_vals, y_vals, 'k--', label='PLA boundary')
    else:
        # vertical line w0 + w1*x = 0 => x = -w0/w1
        if abs(w[1]) > 1e-8:
            x0 = -w[0] / w[1]
            plt.axvline(x=x0, color='k', linestyle='--', label='PLA boundary')

    plt.legend()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    X, y0 = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=1.8, random_state=42)
    y = np.where(y0 == 0, -1, +1)  

    #call and train algorithm
    w, n_updates, epochs_run = pla_train(X, y, max_epochs=1000, alpha=0.1, shuffle=True, random_state=42)

    print(f"PLA finished: updates={n_updates}, epochs={epochs_run}")
    print("Weights (Wo, W1, W2):", w)

    # Plot result
    plot_data_and_boundary(X, y, w, title=f"PLA result — updates={n_updates}, epochs={epochs_run}")