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
	./$(GPU_DIR)/vector.x
	@echo "\n\033[1;38;5mExecuting $(CPU_DIR)/vector.x \033[0m"
	./$(CPU_DIR)/vector.x
	@touch $@

$(TMP)/matmul.tmp: $(GPU_DIR)/matmul.x $(CPU_DIR)/matmul.x | $(TMP)
	@echo "\n\033[1;38;5mExecuting $(GPU_DIR)/matmul.x \033[0m"
	./$(GPU_DIR)/matmul.x
	@echo "\n\033[1;38;5mExecuting $(CPU_DIR)/matmul.x \033[0m"
	./$(CPU_DIR)/matmul.x
	@touch $@

vector-times-gpu:
	@echo "     \033[1;38;5;214mVECTOR (GPU)\033[0m\n$(headers)" && \
	for size in $(VSIZES); do ./$(GPU_DIR)/vector.x $$size 2>$(LOG)/$@.log; done | tee $(DAT)/$@.txt

vector-times-cpu:
	@echo "     \033[1;38;5;214mVECTOR (CPU)\033[0m\n$(headers)" && \
	for size in $(VSIZES); do OMP_NUM_THREADS=32 ./$(CPU_DIR)/vector.x $$size 2>$(LOG)/strong-$@.log; done | tee $(DAT)/strong-$@.txt
	@echo "                    \n$(headerw)" && \
	for thread in $(THREADS); do echo -n "$$thread\t" && OMP_NUM_THREADS=$$thread ./$(CPU_DIR)/vector.x $(vweaksize) 2>$(LOG)/weak-$@.log; done | tee $(DAT)/weak-$@.txt

matmul-times-gpu:
	@echo "     \033[1;38;5;214mMATMUL (GPU)\033[0m\n$(headers)" && \
	for size in $(MSIZES); do ./$(GPU_DIR)/matmul.x $$size 2>$(LOG)/$@.log; done | tee $(DAT)/$@.txt

matmul-times-cpu:
	@echo "     \033[1;38;5;214mMATMUL (CPU)\033[0m\n$(headers)" && \
	for size in $(MSIZES); do OMP_NUM_THREADS=32 ./$(CPU_DIR)/matmul.x $$size 2>$(LOG)/strong-$@.log; done | tee $(DAT)/strong-$@.txt
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

##############################################################################################################
## Regla para ejecutar Weak Scaling
weak-scaling: $(TMP)/weak-scalingvector.tmp $(TMP)/weak-scalingmatmul.tmp
	@echo "\033[38;5;70m\nData has been created!\nPlot by running 'make plot'\033[0m\n"

WEAK_DIR=GPU_weakScaling

# Reglas de compilación para GPU Weak Scaling
$(WEAK_DIR)/matmul.x: $(WEAK_DIR)/mat_mul.cu | $(TMP)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CUDA) $< -o $@

$(WEAK_DIR)/vector.x: $(WEAK_DIR)/vector.cu | $(TMP)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CUDA) $< -o $@

# Reglas para generar archivos temporales GPU Weak Scaling
$(TMP)/weak-scalingvector.tmp: $(WEAK_DIR)/vector.x | $(TMP)
	@$(MAKE) vector-times-weak
	@touch $@

$(TMP)/weak-scalingmatmul.tmp: $(WEAK_DIR)/matmul.x | $(TMP)
	@$(MAKE) matmul-times-weak
	@touch $@

# Define the size and scale_factor parameters in a single array
VECTOR_PARAMS := 100:1 100:2 100:4 100:8 100:16 500:1 500:2 500:4 500:8 500:16
MATMUL_PARAMS := 100:1 100:2 100:4 100:8 100:16 500:1 500:2 500:4 500:8 500:16

header_weak=size\tscale_factor\ttime

# Rule for executing vector with weak scaling
vector-times-weak:
	@echo "     \033[1;38;5;214mVECTOR (GPU)\033[0m\n$(header_weak)" && \
	for param in $(VECTOR_PARAMS); do \
	    size=$$(echo $$param | cut -d: -f1); \
	    scale_factor=$$(echo $$param | cut -d: -f2); \
	    time=$$(./$(WEAK_DIR)/vector.x $$size $$scale_factor 2>$(LOG)/gpu_vector.log); \
	    echo "$$size\t$$scale_factor\t$$time"; \
	done | tee $(DAT)/gpu_vector-times-weak.txt

# Rule for executing matmul with weak scaling
matmul-times-weak:
	@echo "     \033[1;38;5;214mMATMUL (GPU)\033[0m\n$(header_weak)" && \
	for param in $(MATMUL_PARAMS); do \
	    size=$$(echo $$param | cut -d: -f1); \
	    scale_factor=$$(echo $$param | cut -d: -f2); \
	    time=$$(./$(WEAK_DIR)/matmul.x $$size $$scale_factor 2>$(LOG)/gpu_matmul.log); \
	    echo "$$size\t$$scale_factor\t$$time"; \
	done | tee $(DAT)/gpu_matmul-times-weak.txt

######################################################################################################
