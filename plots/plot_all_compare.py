import matplotlib.pyplot as plt
import numpy as np

# Corresponding cell sizes for your timing data
sizes = np.array([10, 100, 1000, 10000]) # adjust to match your actual sizes length

original = np.array([0.042890, 3.881494, 447.790748])
cython = np.array([0.049108, 0.070243, 1.818065, 198.65835])
mpi4py_best_n_10 = np.array([0.049838, 1.150866, 82.121691])
np_vec = np.array([0.024455, 0.251372, 12.062716])
threads_10 = np.array([0.031744, 0.0540481, 1.7610673])

fig, ax = plt.subplots(figsize=(7,5), dpi=300)

ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(sizes[:len(original)], original, marker='d', color='black', label='python')
ax.plot(sizes[:len(cython)], cython, marker='d', color='mediumblue', label='cython')
ax.plot(sizes[:len(mpi4py_best_n_10)], mpi4py_best_n_10, marker='d', color='deepskyblue', label='mpi4py')
ax.plot(sizes[:len(np_vec)], np_vec, marker='d', color='dodgerblue', label='np_vec')
ax.plot(sizes[:len(threads_10)], threads_10, marker='d', color='royalblue', label='numba')

ax.set_xlim(10, 25000)
ax.set_ylim(1e-3, 1000)
ax.set_xlabel('X size (cells)', fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)
ax.legend(frameon=False)

plt.tight_layout(pad=2.5)
plt.show()