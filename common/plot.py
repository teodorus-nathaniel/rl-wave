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

def plot_values_and_trend(ax, values):
    ax.plot(values, label='score per run')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    x = range(len(values))
    ax.legend()

    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), '--', label='trend')
    except:
        pass

def plot_res(values, title='', smooth=50):
    clear_output(wait=True)

    smoothed_values = avg_per_x_element(values, smooth)
    smoothed_values_2 = avg_per_x_element(values, smooth * 2)

    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    f.suptitle(title)

    # plot smoothed values
    plot_values_and_trend(ax[0, 0], smoothed_values)
    plot_values_and_trend(ax[0, 1], smoothed_values_2)

    # Plot the histogram of results
    ax[1, 0].hist(values[-smooth:])
    ax[1, 0].set_xlabel(f'Scores per Last {smooth} Episodes')
    ax[1, 0].set_ylabel('Frequency')
    ax[1, 0].legend()

    # Plot last {smooth} episodes
    last_values = values[-smooth:]
    plot_values_and_trend(ax[1, 1], last_values)

    plt.show()
