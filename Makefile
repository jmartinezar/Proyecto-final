.PHONY: report

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


SIZES=10 100 500 1000 5000 10000 15000 20000 25000 29000
# SIZES=10 100 500 1000 5000

header=size\ttime(s)

GPU_DIR=GPU
CPU_DIR=CPU

# Imprimir el valor de TYPE antes de los ifs
$(info TYPE is $(TYPE))

# Asegúrate de que los directorios existan
$(TMP):
	mkdir -p $(TMP)
$(DAT):
	mkdir -p $(DAT)
$(LOG):
	mkdir -p $(LOG)



all: $(TMP)/$(TYPE)vector.tmp $(TMP)/$(TYPE)matmul.tmp
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
	$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -o $@
endif

$(CPU_DIR)/vector.x: $(CPU_DIR)/vector.cpp | $(TMP)
ifneq ($(filter cpu,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -o $@
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
	@echo "     \033[1;38;5;214mVECTOR (GPU)\033[0m\n$(header)" && \
	for size in $(SIZES); do ./$(GPU_DIR)/vector.x $$size 2>$(LOG)/gpu_$@.log; done | tee $(DAT)/gpu_$@.txt
endif
ifneq ($(filter cpu,$(TYPE)),)
	@echo "     \033[1;38;5;214mVECTOR (CPU)\033[0m\n$(header)" && \
	for size in $(SIZES); do ./$(CPU_DIR)/vector.x $$size 2>$(LOG)/cpu_$@.log; done | tee $(DAT)/cpu_$@.txt
endif

matmul-times:
ifneq ($(filter gpu,$(TYPE)),)
	@echo "     \033[1;38;5;214mMATMUL (GPU)\033[0m\n$(header)" && \
	for size in $(SIZES); do ./$(GPU_DIR)/matmul.x $$size 2>$(LOG)/gpu_$@.log; done | tee $(DAT)/gpu_$@.txt
endif
ifneq ($(filter cpu,$(TYPE)),)
	@echo "     \033[1;38;5;214mMATMUL (CPU)\033[0m\n$(header)" && \
	for size in $(SIZES); do ./$(CPU_DIR)/matmul.x $$size 2>$(LOG)/cpu_$@.log; done | tee $(DAT)/cpu_$@.txt
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
