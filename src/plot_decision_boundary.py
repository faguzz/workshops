# Plotting the decision boundaries
import numpy as np
import matplotlib.pyplot as plt

# Plot the decision boundary
def plot_decision_boundary(X, Y, headers, dims, target_names, clf):
    # Some variables
    n_classes = 3
    plot_colors = "bry"
    plot_step = 0.02
    # Set min and max for two dimensions
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    # Create matrix of steps from min to max
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    # Predict every point from the matrix
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # what?
    Z = Z.reshape(xx.shape)
    # Plot contour
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    # Set labels
    plt.xlabel(headers[dims[0]])
    plt.ylabel(headers[dims[1]])
    plt.axis("tight")
    # Plot train points, colored by class
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(Y == i)
        plt.scatter(X[idx,0], X[idx,1], c=color, label=target_names[i], cmap=plt.cm.Paired)
