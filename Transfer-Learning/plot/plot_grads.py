import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Plotting_alpha_grads(alpha_values, grads):
    num_iterations = len(grads[list(grads.keys())[0]])
    grads = np.array([grads[name] for name in grads]).reshape(num_iterations, -1)
    print(num_iterations)
    gradient_mean = np.mean(grads, axis=1)
    gradient_variance = np.var(grads, axis=1)

    alpha_values = np.array(alpha_values)
    # Plotting the gradients and alpha values
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot gradients of the last layer
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Gradient Value', color='tab:blue')
    for i in grads.shape[1]:
        ax1.plot(grads[:, i], alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot mean of gradients
    ax1.plot(gradient_mean, label='Gradient Mean', color='tab:green')
    ax1.legend(loc='upper left')

    # Create a second y-axis for the variance
    ax2 = ax1.twinx()
    ax2.set_ylabel('Gradient Variance', color='tab:red')
    ax2.plot(gradient_variance, label='Gradient Variance', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    # Plot alpha values
    ax2 = ax1.twinx()
    ax2.set_ylabel('Alpha Value', color='tab:red')
    ax2.plot(alpha_values, label='Alpha', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    plt.title('Gradient Changes in the Last Layer and Alpha Value During Training')
    plt.show()

def Plotting_grads_statics(grads):
    num_iterations = len(grads[list(grads.keys())[0]])
    gradient_data = np.array([grads[name] for name in grads]).reshape(num_iterations, -1)
    print(num_iterations, gradient_data.shape)
    # Calculate mean and variance of the gradients over time
    gradient_mean = np.mean(gradient_data, axis=1)
    gradient_variance = np.var(gradient_data, axis=1)

    # Plotting the gradients, mean, and variance
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot gradients of the last layer
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Gradient Value', color='tab:blue')
    for i in range(gradient_data.shape[1]):
        ax1.plot(gradient_data[:, i], alpha=0.3)  # Individual gradient values with transparency
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot mean of gradients
    ax1.plot(gradient_mean, label='Gradient Mean', color='tab:green')
    ax1.legend(loc='upper left')

    # Create a second y-axis for the variance
    ax2 = ax1.twinx()
    ax2.set_ylabel('Gradient Variance', color='tab:red')
    ax2.plot(gradient_variance, label='Gradient Variance', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    plt.title('Gradient Changes, Mean, and Variance in the Last Layer During Training')
    plt.show()
