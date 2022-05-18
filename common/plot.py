import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


def smooth_values(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def plot_values_and_trend(ax, values, label):
    ax.plot(values, label=label)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    x = range(len(values))
    ax.legend()

    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "--", label="trend")
    except:
        pass


def plot_res(values, title="", smooth=50):
    clear_output(wait=True)

    smoothed_values = smooth_values(values, smooth)
    smoothed_values_2 = smooth_values(values, smooth * 2)

    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    f.suptitle(title)

    # plot smoothed values
    plot_values_and_trend(ax[0, 0], smoothed_values, f"score per {smooth} run")
    plot_values_and_trend(ax[0, 1], smoothed_values_2, f"score per {smooth * 2} run")

    # Plot the histogram of results
    ax[1, 0].hist(values[-smooth:])
    ax[1, 0].set_xlabel(f"Scores per Last {smooth} Episodes")
    ax[1, 0].set_ylabel("Frequency")
    ax[1, 0].legend()

    # Plot last {smooth} episodes
    last_values = values[-smooth:]
    plot_values_and_trend(ax[1, 1], last_values, f"score last {smooth} run")

    plt.show()
