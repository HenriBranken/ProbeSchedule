import matplotlib.pyplot as plt
import numpy as np

n = 5

x = np.arange(n)
y = np.sin(np.linspace(-3, 3, n))
xlabels = ['Ticklabel %i' % i for i in range(n)]

fig, axs = plt.subplots(1, 3, figsize=(12, 3))

ha = ['right', 'center', 'left']

for n, ax in enumerate(axs):
    ax.plot(x, y, 'o-')
    ax.set_title(ha[n])
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=40, ha=ha[n])
plt.show()
