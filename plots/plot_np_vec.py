import matplotlib.pyplot as plt
import numpy as np

sizes = np.array([10, 100, 1000, 10000])

original = np.array([0.042890, 3.881494, 447.790748])
np_vec = np.array([0.024455, 0.251372, 12.062716])

fig, ax = plt.subplots(figsize=(7,5), dpi=300)

ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(sizes[:len(original)], original, marker='d', color='black', label='python')
ax.plot(sizes[:len(np_vec)], np_vec, marker='d', color='dodgerblue', label='np_vec')

ax.set_xlim(10, 25000)
ax.set_ylim(1e-3, 1000)
ax.set_xlabel('X size (cells)', fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)
ax.legend(frameon=False)

plt.tight_layout(pad=2.5)
plt.show()