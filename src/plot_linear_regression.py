# Importing matplotlib's library
import matplotlib.pyplot as plt

# Function for plotting Linear Regression
def plot_linear_regression(train_data, train_target, test_data, test_target):
    # plot train data as black dots
    plt.scatter(train_data, train_target, color='black')
    # plot train data as red dots
    plt.scatter(test_data, test_target, color='red')
    # plot the linear regression model as blue line
    plt.plot(test_data, lr.predict(test_data), color='blue')
    plt.show()
