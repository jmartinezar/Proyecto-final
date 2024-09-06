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


MSIZES=10 100 1000 10000 100000 1000000 10000000 100000000
VSIZES=1000 10000 1000000 10000000 100000000 1000000000 10000000000 100000000000
THREADS=8 16 32 64 128 256 512 1024
# SIZES=10 100 500 1000 5000


numthreads=256
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


all: $(TMP)/vector.tmp $(TMP)/matmul.tmp

# compilacion de matmul GPU
$(GPU_DIR)/matmul.x: $(GPU_DIR)/mat_mul.cu | $(TMP)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CUDA) $< -o $@

# compilacion de vector GPU
$(GPU_DIR)/vector.x: $(GPU_DIR)/vector.cu | $(TMP)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CUDA) $< -o $@

# compilacion de matmul CPU
$(CPU_DIR)/matmul.x: $(CPU_DIR)/mat_mul.cpp | $(TMP)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -fopenmp -o $@

# compilacion de vector CPU
$(CPU_DIR)/vector.x: $(CPU_DIR)/vector.cpp | $(TMP)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -fopenmp -o $@

$(TMP)/vector.tmp: $(GPU_DIR)/vector.x $(CPU_DIR)/vector.x | $(TMP)
	@echo "\n\033[1;38;5mExecuting $(GPU_DIR)/vector.x \033[0m"
	@$(MAKE) vector-times-gpu
	@echo "\n\033[1;38;5mExecuting $(CPU_DIR)/vector.x \033[0m"
	@$(MAKE) vector-times-cpu
	@touch $@

$(TMP)/matmul.tmp: $(GPU_DIR)/matmul.x $(CPU_DIR)/matmul.x | $(TMP)
	@echo "\n\033[1;38;5mExecuting $(GPU_DIR)/matmul.x \033[0m"
	@$(MAKE) matmul-times-gpu
	@echo "\n\033[1;38;5mExecuting $(CPU_DIR)/matmul.x \033[0m"
	@$(MAKE) matmul-times-cpu
	@touch $@

vector-times-gpu:
	@echo "     \033[1;38;5;214mVECTOR (GPU)\033[0m\n$(headers)" && \
	for size in $(VSIZES); do ./$(GPU_DIR)/vector.x $$size $(numthreads) 2>$(LOG)/strong-$@.log; done | tee $(DAT)/strong-$@.txt
	@echo "                    \n$(headerw)" && \
	for thread in $(THREADS); do echo -n "$$thread\t" && ./$(GPU_DIR)/vector.x $(vweaksize) $$thread 2>$(LOG)/weak-$@.log; done | tee $(DAT)/weak-$@.txt

vector-times-cpu:
	@echo "     \033[1;38;5;214mVECTOR (CPU)\033[0m\n$(headers)" && \
	for size in $(VSIZES); do OMP_NUM_THREADS=$(numthreads) ./$(CPU_DIR)/vector.x $$size 2>$(LOG)/strong-$@.log; done | tee $(DAT)/strong-$@.txt
	@echo "                    \n$(headerw)" && \
	for thread in $(THREADS); do echo -n "$$thread\t" && OMP_NUM_THREADS=$$thread ./$(CPU_DIR)/vector.x $(vweaksize) 2>$(LOG)/weak-$@.log; done | tee $(DAT)/weak-$@.txt

matmul-times-gpu:
	@echo "     \033[1;38;5;214mMATMUL (GPU)\033[0m\n$(headers)" && \
	for size in $(MSIZES); do ./$(GPU_DIR)/matmul.x $$size $(numthreads) 2>$(LOG)/strong-$@.log; done | tee $(DAT)/strong-$@.txt
	@echo "                    \n$(headerw)" && \
	for thread in $(THREADS); do echo -n "$$thread\t" && ./$(GPU_DIR)/matmul.x $(mweaksize) $$thread 2>$(LOG)/weak-$@.log; done | tee $(DAT)/weak-$@.txt

matmul-times-cpu:
	@echo "     \033[1;38;5;214mMATMUL (CPU)\033[0m\n$(headers)" && \
	for size in $(MSIZES); do OMP_NUM_THREADS=$(numthreads) ./$(CPU_DIR)/matmul.x $$size 2>$(LOG)/strong-$@.log; done | tee $(DAT)/strong-$@.txt
	@echo "                    \n$(headerw)" && \
	for thread in $(THREADS); do echo -n "$$thread\t" && OMP_NUM_THREADS=$$thread ./$(CPU_DIR)/matmul.x $(mweaksize) 2>$(LOG)/weak-$@.log; done | tee $(DAT)/weak-$@.txt


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
