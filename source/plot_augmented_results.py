import matplotlib.pyplot as plt
from os.path import isfile
import pickle
import numpy as np
from matplotlib import colors as mcolors

def plot_it_all():
    plt.figure()
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
    count = 0
    for batchsize in [50, 100, 200, 500, 2000]:
        for augmented in [0, 1]:
            returns = []
            for j in range(1000):
                filename = "../data/new_run_num_%d_augment_%d_batchsize_%d.pkl" % (j, int(augmented), batchsize)
                if isfile(filename):
                    data = pickle.load(open(filename, "rb"))[-50:]
                    returns.append(data)
            mean = np.mean(returns, axis=0)
            variance = np.std(returns, axis=0)
            if augmented:
                linestyle = '--'
            else:
                linestyle = '-'
            plt.plot(mean,
                     linewidth=2.0,
                     linestyle=linestyle,
                     color = colors[count],
                     label="augmented %d, batch %d" % (augmented, batchsize))
            plt.fill_between(np.arange(len(mean)), mean - variance, mean + variance, alpha=0.3,
                                facecolor=colors[count])
        count += 1
    plt.grid()
    plt.legend()
    plt.show(True)


plot_it_all()