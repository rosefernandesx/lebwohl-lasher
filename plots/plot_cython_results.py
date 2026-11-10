import matplotlib.pyplot as plt
import numpy as np

# Corresponding cell sizes for your timing data
sizes = np.array([10, 100, 1000, 10000]) # adjust to match your actual sizes length

original = np.array([0.042890, 3.881494, 447.790748])
cython = np.array([0.049108, 0.070243, 1.818065, 198.65835])

fig, ax = plt.subplots(figsize=(7,5), dpi=300)

ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(sizes[:len(original)], original, marker='d', color='black', label='python')
ax.plot(sizes[:len(cython)], cython, marker='d', color='mediumblue', label='cython')

ax.set_xlim(10, 25000)
ax.set_ylim(1e-3, 1000)
ax.set_xlabel('X size (cells)', fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)
ax.legend(frameon=False)

plt.tight_layout(pad=2.5)
plt.show()