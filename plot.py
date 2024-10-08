import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import AutoMinorLocator, LogLocator

fm.fontManager.addfont("aux/Garuda.ttf")
fm.fontManager.addfont("aux/Garuda-Bold.ttf")

plt.rcParams["font.family"] = "Garuda"
size = 13
plt.rcParams["font.size"] = size

# ================================ data ===================================

strong_cpu_matmul = np.genfromtxt("data/metrics/strong-matmul-times-cpu-metrics.txt")
strong_gpu_matmul = np.genfromtxt("data/metrics/strong-matmul-times-gpu-metrics.txt")
strong_cpu_vector = np.genfromtxt("data/metrics/strong-vector-times-cpu-metrics.txt")
strong_gpu_vector = np.genfromtxt("data/metrics/strong-vector-times-gpu-metrics.txt")
weak_cpu_matmul = np.genfromtxt("data/raw/weak-matmul-times-cpu.txt")
weak_gpu_matmul = np.genfromtxt("data/raw/weak-matmul-times-gpu.txt")
weak_cpu_vector = np.genfromtxt("data/raw/weak-vector-times-cpu.txt")
weak_gpu_vector = np.genfromtxt("data/raw/weak-vector-times-gpu.txt")

# =============================== plot ====================================

# -------------- matrix multiplication --------------
nthM = np.arange(strong_cpu_matmul[0,0], strong_cpu_matmul[-1,0], 1)
nthgM = np.arange(strong_gpu_matmul[0,0], strong_gpu_matmul[-5,0], 1)

figM, axM = plt.subplot_mosaic([['top', 'top'], ['l1', 'r1'], ['l2', 'r2']], figsize = (10, 12))

figM.suptitle("Matrix multiplication parallel metrics", fontweight='bold')

axM['top'].set_title("Weak scaling")
axM['top'].set_xlabel("Num threads")
axM['top'].set_ylabel("Execution time (logscale) [s]")
axM['top'].set_yscale('log')
axM['top'].plot(weak_cpu_matmul[:,0], weak_cpu_matmul[:,2], '-^', label = 'CPU execution', c = 'k', mec = 'k', mfc = (0,0,0,0), ms = 8)
axM['top'].plot(weak_gpu_matmul[:,0], weak_gpu_matmul[:,2], '-s', label = 'GPU execution', c = '#c20000', mec = '#c20000', mfc = (0,0,0,0), ms = 8)
axM['top'].yaxis.set_major_locator(LogLocator(base=10, numticks=15))
axM['top'].yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=10))
axM['top'].tick_params(axis='y', which='major', length=10, width=.3)
axM['top'].tick_params(axis='y', which='minor', length=5, width=.3)
axM['top'].xaxis.set_minor_locator(AutoMinorLocator(5))
axM['top'].grid(which = 'both', lw = .3)

axM['top'].legend()

figM.text(.485,.635, "Speedup", fontweight= 'bold', fontsize = size + 3)

axM['l1'].set_title("\nGPU", fontweight='bold')
axM['l1'].set_ylabel("Speedup (linscale)", c='#c20000')
axM['l1'].tick_params(axis= 'y', rotation=90)
axM['l1'].plot(strong_gpu_matmul[:-4,0], strong_gpu_matmul[:-4,2], 's', c = '#c20000', mec = '#c20000', mfc = (0,0,0,0), ms = 8)
axM['l1'].plot(nthgM, nthgM, c = '#edc2c2', ls = '--', lw = 1, label = '$y=x$')
axM['l1'].grid(which = 'both', lw = .3)

axML=axM['l1'].twinx()
axML.set_ylabel("Speedup (logscale)", c='#0009c2')
axML.set_yscale('log')
axML.plot(strong_gpu_matmul[:-4,0], strong_gpu_matmul[:-4,2], 's', c = '#0009c2', mec = '#0009c2', mfc = (0,0,0,0), ms = 8)
axML.plot(nthgM, nthgM, c = '#cfd0ee', ls = '--', lw = 1, label = '$\\log y= x$')
axML.grid(which = 'both', lw = .3)

axML.legend(loc='upper left', bbox_to_anchor = (0, .86))
axM['l1'].legend()

axM['l1'].set_xlabel("Num threads")

axM['r1'].set_title("\nCPU", fontweight='bold')
axM['r1'].set_ylabel("Speedup")
axM['r1'].plot(strong_cpu_matmul[:,0], strong_cpu_matmul[:,2], '^', c = 'k', mec = 'k', mfc = (0,0,0,0), ms = 8)
axM['r1'].plot(nthM, nthM, c = '#bdbdbd', ls = '--', lw = 1, label = '$y=x$')
axM['r1'].grid(which = 'both', lw = .3)

axM['r1'].legend()

axM['r1'].set_xlabel("Num threads")

figM.text(.48,.315, "Efficiency", fontweight= 'bold', fontsize = size + 3)

axM['l2'].set_title("\nGPU", fontweight='bold')
axM['l2'].set_ylabel("Efficiency")
axM['l2'].axhline(0.7, ls='--', c='#bdbdbd', label = '$E_{ff}=0.7$')
axM['l2'].plot(strong_gpu_matmul[:-4,0], strong_gpu_matmul[:-4,3], 's', c = '#c20000', mec = '#c20000', mfc = (0,0,0,0), ms = 8)
axM['l2'].grid(which = 'both', lw = .3)

axM['l2'].legend()

axM['l2'].set_xlabel("Num threads")

axM['r2'].set_title("\nCPU", fontweight='bold')
axM['r2'].set_ylabel("Efficiency")
axM['r2'].axhline(0.7, ls='--', c='#bdbdbd', label = '$E_{ff}=0.7$')
axM['r2'].plot(strong_cpu_matmul[:,0], strong_cpu_matmul[:,3], '^', c = 'k', mec = 'k', mfc = (0,0,0,0), ms = 8)
axM['r2'].grid(which = 'both', lw = .3)

axM['r2'].legend()

axM['r2'].set_xlabel("Num threads")

figM.tight_layout()
figM.savefig("figs/matmul-scaling.pdf")
figM.savefig("figs/matmul-scaling.png", dpi=300)

# ------------------- vector sum -----------------------
nth = np.arange(strong_cpu_vector[0,0], strong_cpu_vector[-1,0], 1)
nthg = np.arange(strong_gpu_vector[0,0], strong_gpu_vector[-4,0], 1)

figV, axV = plt.subplot_mosaic([['top', 'top'], ['l1', 'r1'], ['l2', 'r2']], figsize = (10, 12))

figV.suptitle("Vector sum parallel metrics", fontweight='bold')

axV['top'].set_title("Weak scaling")
axV['top'].set_xlabel("Num threads")
axV['top'].set_ylabel("Execution time (logscale) [s]")
# axV['top'].set_xscale('log')
axV['top'].set_yscale('log')
axV['top'].plot(weak_cpu_vector[:,0], weak_cpu_vector[:,2], '-^', label = 'CPU execution', c = 'k', mec = 'k', mfc = (0,0,0,0), ms = 8)
axV['top'].plot(weak_gpu_vector[:,0], weak_gpu_vector[:,2], '-s', label = 'GPU execution', c = '#c20000', mec = '#c20000', mfc = (0,0,0,0), ms = 8)
axV['top'].yaxis.set_major_locator(LogLocator(base=10, numticks=15))
axV['top'].yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=10))
# axV['top'].xaxis.set_major_locator(LogLocator(base=10, numticks=15))
# axV['top'].xaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=10))
axV['top'].tick_params(axis='both', which='major', length=10, width=.3)
axV['top'].tick_params(axis='both', which='minor', length=5, width=.3)
axV['top'].grid(which = 'both', lw = .3)

axV['top'].legend()

figV.text(.485,.635, "Speedup", fontweight= 'bold', fontsize = size + 3)

axV['l1'].set_title("\nGPU", fontweight='bold')
axV['l1'].set_ylabel("Speedup")
axV['l1'].plot(strong_gpu_vector[:-4,0], strong_gpu_vector[:-4,2], 's', c = '#c20000', mec = '#c20000', mfc = (0,0,0,0), ms = 8)
axV['l1'].plot(nthg, nthg, c = '#bdbdbd', ls = '--', lw = 1, label = '$y=x$')
axV['l1'].grid(which = 'both', lw = .3)

axV['l1'].legend()
axV['l1'].set_xlabel("Num threads")

axV['r1'].set_title("\nCPU", fontweight='bold')
axV['r1'].set_ylabel("Speedup")
axV['r1'].plot(strong_cpu_vector[:,0], strong_cpu_vector[:,2], '^', c = 'k', mec = 'k', mfc = (0,0,0,0), ms = 8)
axV['r1'].plot(nth, nth, c = '#bdbdbd', ls = '--', lw = 1, label = '$y=x$')
axV['r1'].grid(which = 'both', lw = .3)

axV['r1'].legend()
axV['r1'].set_xlabel("Num threads")

figV.text(.48,.315, "Efficiency", fontweight= 'bold', fontsize = size + 3)

axV['l2'].set_title("\nGPU", fontweight='bold')
axV['l2'].set_ylabel("Efficiency")
axV['l2'].axhline(0.7, ls='--', c='#bdbdbd', label='$E_{ff} = 0.7$')
axV['l2'].plot(strong_gpu_vector[:-4,0], strong_gpu_vector[:-4,3], 's', c = '#c20000', mec = '#c20000', mfc = (0,0,0,0), ms = 8)
axV['l2'].grid(which = 'both', lw = .3)

axV['l2'].legend()
axV['l2'].set_xlabel("Num threads")

axV['r2'].set_title("\nCPU", fontweight='bold')
axV['r2'].set_ylabel("Efficiency")
axV['r2'].axhline(0.7, ls='--', c='#bdbdbd', label='$E_{ff} = 0.7$')
axV['r2'].plot(strong_cpu_vector[:,0], strong_cpu_vector[:,3], '^', c = 'k', mec = 'k', mfc = (0,0,0,0), ms = 8)
axV['r2'].grid(which = 'both', lw = .3)

axV['r2'].legend()
axV['r2'].set_xlabel("Num threads")

figV.tight_layout()
figV.savefig("figs/vector-scaling.pdf")
figV.savefig("figs/vector-scaling.png", dpi=300)
