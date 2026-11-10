import matplotlib.pyplot as plt
import numpy as np

sizes = np.array([10, 100, 1000, 10000])

original = np.array([0.042890, 3.881494, 447.790748])
threads_2 = np.array([0.033087, 0.054180, 1.835662])
threads_4 = np.array([0.033953, 0.054822, 1.7817573])
threads_6 = np.array([0.033418, 0.054968, 1.7601143])
threads_8 = np.array([0.033013, 0.053140, 1.9565833])
threads_10 = np.array([0.031744, 0.0540481, 1.7610673])

fig, ax = plt.subplots(figsize=(7,5), dpi=300)

ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(sizes[:len(original)], original, marker='d', color='black', label='python')
ax.plot(sizes[:len(threads_2)], threads_2, marker='d', color='navy', label='2 threads')
ax.plot(sizes[:len(threads_4)], threads_4, marker='d', color='mediumblue', label='4 threads')
ax.plot(sizes[:len(threads_6)], threads_6, marker='d', color='royalblue', label='6 threads')
ax.plot(sizes[:len(threads_8)], threads_8, marker='d', color='dodgerblue', label='8 threads')
ax.plot(sizes[:len(threads_10)], threads_10, marker='d', color='deepskyblue', label='10 threads')

ax.set_xlim(10, 25000)
ax.set_ylim(1e-3, 1000)
ax.set_xlabel('X size (cells)', fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)
ax.legend(frameon=False)

plt.tight_layout(pad=2.5)
plt.show()