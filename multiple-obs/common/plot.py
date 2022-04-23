import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np


def plot_res(values, title=""):
    """Plot the reward curve and histogram of results over time."""
    # Update the window after each episode
    clear_output(wait=True)

    # Define the figure
    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    f.suptitle(title)
    ax[0, 0].plot(values, label="score per run")
    ax[0, 0].set_xlabel("Episodes")
    ax[0, 0].set_ylabel("Reward")
    x = range(len(values))
    ax[0, 0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0, 0].plot(x, p(x), "--", label="trend")
    except:
        print("")

    # Plot the histogram of results
    ax[0, 1].hist(values[-50:])
    ax[0, 1].set_xlabel("Scores per Last 50 Episodes")
    ax[0, 1].set_ylabel("Frequency")
    ax[0, 1].legend()

    # Plot last 100
    ax[1, 0].plot(values[-100:])
    ax[1, 0].set_xlabel("Episodes")
    ax[1, 0].set_ylabel("Reward")
    ax[1, 0].legend()
    plt.show()
