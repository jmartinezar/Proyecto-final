.PHONY: report
.DEFAULT_GOAL := all

TYPE ?= gpu
COMPILER_CUDA=nvcc
COMPILER_CPP=g++
CXXFLAGS_CPP=-I$(HOME)/eigen-3.4.0 -O2
DAT=data
LOG=logs
TMP=tmp
REP=report
# vector times plot seems to representate random data, this may be due to the current range of 
# size and the complexity of vector-sum opperation. This range need to be checked


MSIZES=10 100 500 1000 5000 10000 15000 20000 25000 29000
VSIZES=1000 10000 1000000 10000000 100000000 500000000
THREADS=1 2 4 8 16 32 40 48 56 64
# SIZES=10 100 500 1000 5000

mweaksize=20000
vweaksize=10000000
headers=size\ttime(s)
headerw=threads\tsize\ttime(s)

GPU_DIR=GPU
CPU_DIR=CPU

# Imprimir el valor de TYPE antes de los ifs
# $(info TYPE is $(TYPE))

# Asegúrate de que los directorios existan
$(TMP):
	mkdir -p $(TMP)
$(DAT):
	mkdir -p $(DAT)
$(LOG):
	mkdir -p $(LOG)

all:
	$(info TYPE is $(TYPE))
	@$(MAKE) exec TYPE=gpu
	$(info TYPE is $(TYPE))
	@$(MAKE) exec TYPE=cpu

exec: $(TMP)/$(TYPE)vector.tmp $(TMP)/$(TYPE)matmul.tmp
	@echo "\033[38;5;70m\nData has been created!\nPlot by running 'make plot'\033[0m\n"

# Reglas de compilación para GPU
$(GPU_DIR)/matmul.x: $(GPU_DIR)/mat_mul.cu | $(TMP)
ifneq ($(filter gpu,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CUDA) $< -o $@
endif

$(GPU_DIR)/vector.x: $(GPU_DIR)/vector.cu | $(TMP)
ifneq ($(filter gpu,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CUDA) $< -o $@
endif

# Reglas de compilación para CPU
$(CPU_DIR)/matmul.x: $(CPU_DIR)/mat_mul.cpp | $(TMP)
ifneq ($(filter cpu,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -fopenmp -o $@
endif

$(CPU_DIR)/vector.x: $(CPU_DIR)/vector.cpp | $(TMP)
ifneq ($(filter cpu,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -fopenmp -o $@
endif

# Reglas para generar archivos temporales
$(TMP)/$(TYPE)vector.tmp: $(GPU_DIR)/vector.x $(CPU_DIR)/vector.x | $(TMP)
ifneq ($(filter gpu,$(TYPE)),)
	@$(MAKE) vector-times
endif
ifneq ($(filter cpu,$(TYPE)),)
	@$(MAKE) vector-times
endif
	@touch $@

$(TMP)/$(TYPE)matmul.tmp: $(GPU_DIR)/matmul.x $(CPU_DIR)/matmul.x | $(TMP)
ifneq ($(filter gpu,$(TYPE)),)
	@$(MAKE) matmul-times
endif
ifneq ($(filter cpu,$(TYPE)),)
	@$(MAKE) matmul-times
endif
	@touch $@

vector-times:
ifneq ($(filter gpu,$(TYPE)),)
	@echo "     \033[1;38;5;214mVECTOR (GPU)\033[0m\n$(headers)" && \
	for size in $(VSIZES); do ./$(GPU_DIR)/vector.x $$size 2>$(LOG)/gpu_$@.log; done | tee $(DAT)/gpu_$@.txt
endif
ifneq ($(filter cpu,$(TYPE)),)
	@echo "     \033[1;38;5;214mVECTOR (CPU)\033[0m\n$(headers)" && \
	for size in $(VSIZES); do OMP_NUM_THREADS=32 ./$(CPU_DIR)/vector.x $$size 2>$(LOG)/cpu_$@.log; done | tee $(DAT)/strong_cpu_$@.txt
	@echo "                    \n$(headerw)" && \
	for thread in $(THREADS); do echo -n "$$thread\t" && OMP_NUM_THREADS=$$thread ./$(CPU_DIR)/vector.x $(vweaksize) 2>$(LOG)/cpu_$@.log; done | tee $(DAT)/weak_cpu_$@.txt
endif

matmul-times:
ifneq ($(filter gpu,$(TYPE)),)
	@echo "     \033[1;38;5;214mMATMUL (GPU)\033[0m\n$(headers)" && \
	for size in $(MSIZES); do ./$(GPU_DIR)/matmul.x $$size 2>$(LOG)/gpu_$@.log; done | tee $(DAT)/gpu_$@.txt
endif
ifneq ($(filter cpu,$(TYPE)),)
	@echo "     \033[1;38;5;214mMATMUL (CPU)\033[0m\n$(headers)" && \
	for size in $(MSIZES); do OMP_NUM_THREADS=32 ./$(CPU_DIR)/matmul.x $$size 2>$(LOG)/cpu_$@.log; done | tee $(DAT)/strong_cpu_$@.txt
	@echo "                    \n$(headerw)" && \
	for thread in $(THREADS); do echo -n "$$thread\t" && OMP_NUM_THREADS=$$thread ./$(CPU_DIR)/matmul.x $(mweaksize) 2>$(LOG)/cpu_$@.log; done | tee $(DAT)/weak_cpu_$@.txt
endif


plot: plot.py 
	python3 $< ${TYPE}
	@echo "-------------------------------------------------"
	@echo "Figures are saved to 'figs' directory.          |"
	@echo "-------------------------------------------------"

report:
	pdflatex --interaction=batchmode -output-directory=$(REP)/ Entrega-1_PF-HPC-G3.tex

execs-clean:
	rm **/*.x

clean:
	rm figs/* $(DAT)/* $(LOG)/* $(TMP)/* $(REP)/* **/*.x
