import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

def avg_per_x_element(data, x=10):
    avg = []
    sum_data = 0
    i = 0
    for i, el in enumerate(data):
        sum_data += el
        if i > 0 and i % x == 0:
            avg.append(sum_data / x)
            sum_data = 0

    if i % x != 0:
        avg.append(sum_data / (i % x))

    return avg

def plot_res(values, title=""):
    """Plot the reward curve and histogram of results over time."""
    # Update the window after each episode
    clear_output(wait=True)

    smoothed_values = avg_per_x_element(values, 50)

    # Define the figure
    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    f.suptitle(title)
    ax[0, 0].plot(smoothed_values, label="score per run")
    ax[0, 0].set_xlabel("Episodes")
    ax[0, 0].set_ylabel("Reward")
    x = range(len(smoothed_values))
    ax[0, 0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, smoothed_values, 1)
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
    last_100 = values[-100:]
    ax[1, 0].plot(last_100)
    ax[1, 0].set_xlabel("Episodes")
    ax[1, 0].set_ylabel("Reward")
    ax[1, 0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, last_100, 1)
        p = np.poly1d(z)
        ax[1, 0].plot(x, p(x), "--", label="trend")
    except:
        print("")
    plt.show()
