import matplotlib.pyplot as plt
import numpy as np

# Corresponding cell sizes for your timing data
sizes = np.array([10, 100, 1000, 10000]) # adjust to match your actual sizes length

original = np.array([0.042890, 3.881494, 447.790748])
n_2 = np.array([0.046526, 4.033525, 442.794151])
n_4 = np.array([0.024082, 1.517326, 156.431936])
n_6 = np.array([0.025248, 1.406883, 119.792807])
n_8 = np.array([0.02930, 1.172695, 95.958538])
n_10 = np.array([0.049838, 1.150866, 82.121691])

fig, ax = plt.subplots(figsize=(7,5), dpi=300)

plt.rcParams['lines.linewidth'] = 0.5

ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(sizes[:len(original)], original, marker='d', color='black', label='python')
ax.plot(sizes[:len(n_2)], n_2, marker='d', color='navy', label='2 workers')
ax.plot(sizes[:len(n_4)], n_4, marker='d', color='mediumblue', label='4 workers')
ax.plot(sizes[:len(n_6)], n_6, marker='d', color='royalblue', label='4 workers')
ax.plot(sizes[:len(n_8)], n_8, marker='d', color='dodgerblue', label='8 workers')
ax.plot(sizes[:len(n_10)], n_10, marker='d', color='deepskyblue', label='10 workers')

ax.set_xlim(10, 25000)
ax.set_ylim(1e-3, 1000)
ax.set_xlabel('X size (cells)', fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)
ax.legend(frameon=False)

plt.tight_layout(pad=2.5)
plt.show()