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

MSIZES=10 100 500 1000 5000 10000 15000 20000 25000 29000
VSIZES=1000 10000 50000 1000000 5000000 10000000 50000000 100000000 
THREADS=1 2 4 8 14 16 22 28 32 40 48 56 64

numthreads=32
mstrongsize=10000
vstrongsize=10000000
headerw=size\ttime(s)
headers=threads\tsize\ttime(s)

GPU_DIR=GPU
CPU_DIR=CPU
OUTF=en_US.UTF-8

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
	@echo "     \033[1;38;5;214mVECTOR (GPU)\033[0m\n$(headerw)" && \
	for size in $(VSIZES); do ./$(GPU_DIR)/vector.x $$size $(numthreads) 2>$(LOG)/weak-$@.log; done | tee $(DAT)/raw/weak-$@.txt
	@echo "                    \n$(headers)" && \
	for thread in $(THREADS); do echo -n "$$thread\t" && ./$(GPU_DIR)/vector.x $(vstrongsize) $$thread 2>$(LOG)/strong-$@.log; done | tee $(DAT)/raw/strong-$@.txt
	LC_NUMERIC=$(OUTF) awk 'FNR == 1 {val=$$3} {print $$1, $$3, val/$$3, val/$$3/$$1}' $(DAT)/raw/strong-$@.txt > $(DAT)/metrics/strong-$@-metrics.txt

vector-times-cpu:
	@echo "     \033[1;38;5;214mVECTOR (CPU)\033[0m\n$(headerw)" && \
	for size in $(VSIZES); do OMP_NUM_THREADS=$(numthreads) ./$(CPU_DIR)/vector.x $$size 2>$(LOG)/weak-$@.log; done | tee $(DAT)/raw/weak-$@.txt
	@echo "                    \n$(headers)" && \
	for thread in $(THREADS); do echo -n "$$thread\t" && OMP_NUM_THREADS=$$thread ./$(CPU_DIR)/vector.x $(vstrongsize) 2>$(LOG)/strong-$@.log; done | tee $(DAT)/raw/strong-$@.txt
	LC_NUMERIC=$(OUTF) awk 'FNR == 1 {val=$$3} {print $$1, $$3, val/$$3, val/$$3/$$1}' $(DAT)/raw/strong-$@.txt > $(DAT)/metrics/strong-$@-metrics.txt

matmul-times-gpu:
	@echo "     \033[1;38;5;214mMATMUL (GPU)\033[0m\n$(headerw)" && \
	for size in $(MSIZES); do ./$(GPU_DIR)/matmul.x $$size $(numthreads) 2>$(LOG)/weak-$@.log; done | tee $(DAT)/raw/weak-$@.txt
	@echo "                    \n$(headers)" && \
	for thread in $(THREADS); do echo -n "$$thread\t" && ./$(GPU_DIR)/matmul.x $(mstrongsize) $$thread 2>$(LOG)/strong-$@.log; done | tee $(DAT)/raw/strong-$@.txt
	LC_NUMERIC=$(OUTF) awk 'FNR == 1 {val=$$3} {print $$1*$$1, $$3, val/$$3, val/$$3/($$1*$$1)}' $(DAT)/raw/strong-$@.txt > $(DAT)/metrics/strong-$@-metrics.txt

matmul-times-cpu:
	@echo "     \033[1;38;5;214mMATMUL (CPU)\033[0m\n$(headerw)" && \
	for size in $(MSIZES); do OMP_NUM_THREADS=$(numthreads) ./$(CPU_DIR)/matmul.x $$size 2>$(LOG)/weak-$@.log; done | tee $(DAT)/raw/weak-$@.txt
	@echo "                    \n$(headers)" && \
	for thread in $(THREADS); do echo -n "$$thread\t" && OMP_NUM_THREADS=$$thread ./$(CPU_DIR)/matmul.x $(mstrongsize) 2>$(LOG)/strong-$@.log; done | tee $(DAT)/raw/strong-$@.txt
	LC_NUMERIC=$(OUTF) awk 'FNR == 1 {val=$$3} {print $$1*$$1, $$3, val/$$3, val/$$3/($$1*$$1)}' $(DAT)/raw/strong-$@.txt > $(DAT)/metrics/strong-$@-metrics.txt

plot: plot.py 
	python3 $<
	@echo "-------------------------------------------------"
	@echo "Figures are saved to 'figs' directory.          |"
	@echo "-------------------------------------------------"

report:
	pdflatex --interaction=batchmode -output-directory=$(REP)/ Entrega-1_PF-HPC-G3.tex

execs-clean:
	rm **/*.x

clean:
	rm figs/* $(DAT)/* $(LOG)/* $(TMP)/* $(REP)/* **/*.x
