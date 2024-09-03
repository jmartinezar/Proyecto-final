import matplotlib.pyplot as plt
import numpy as np
import sys

# --------------------- Read data ----------------------
TYPE = sys.argv[1].strip().lower()
data_vector = np.loadtxt(f'data/{TYPE}_vector-times.txt')
data_matmul = np.loadtxt(f'data/{TYPE}_matmul-times.txt')

upperType = "GPU" if TYPE == "gpu" else "CPU"
library = "CUDA" if TYPE == "gpu" else "EIGEN/DENSE"

# ---------------------- Plotting ----------------------
# matmul
nologcolor = '#1791E0'
logcolor = '#B30000'    

fig, ax = plt.subplots(sharex = True, figsize = (6,5))
fig.suptitle(f"{upperType} matrix multiplication time scaling using {library}", fontweight="bold", fontsize=13)

ax.set_xlabel("Matrix size", size=13)
ax.set_ylabel(f'{upperType} wall time [s]', size=13, color = nologcolor)
# ax.set_xscale('log')
ax.plot(data_matmul[:,0], data_matmul[:,1], marker = '*', ms = 9, mfc = 'none', mew = 1.0, lw = 1.2, c = nologcolor)

ax2 = ax.twinx()
ax2.set_ylabel(f'{upperType} wall time [s](logscale)', size=13, color = logcolor)
ax2.set_yscale('log')
# ax2.set_xscale('log')
ax2.plot(data_matmul[:,0], data_matmul[:,1], marker = 'o', mfc = 'none', mew = 1.0, ms = 7.5, lw = 1.2, c = logcolor)
plt.tight_layout()
plt.savefig("figs/matmul-strong.png")

# vector
# fig, ax = plt.subplots(sharey = True, figsize = (6,5))
# fig.suptitle(f"{upperType} matrix multiplication time scaling using {library}", fontweight="bold", fontsize=13)
# fig.supylabel(f"{upperType} wall time [s]", size=13)

# ax.set_xlabel('Vector size', size=13, color = '#2a088e')
# ax.plot(data_vector[:,0], data_vector[:,1], marker = '*', ms = 8.5, color = '#2a088e')

# ax2 = ax.twiny()
# ax2.set_xlabel('Vector size (logscale)', size=13, color = '#f4a40a')
# ax2.set_xscale('log')
# ax2.plot(data_vector[:,0], data_vector[:,1], marker = 'o', color = '#f4a40a')
# plt.tight_layout()
# plt.savefig("figs/vector-strong.pdf")

# plot both at once (not useful in server)
# plt.show()