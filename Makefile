.PHONY: report

COMPILER_CUDA=nvcc
COMPILER_CPP=g++
CXXFLAGS_CPP=-I/usr/include/eigen3 -O2
DAT=data
LOG=logs
TMP=tmp
REP=report
# vector times plot seems to representate random data, this may be due to the current range of 
# size and the complexity of vector-sum opperation. This range need to be checked

TYPE ?= gpu

SIZES=10 100 500 1000 5000 10000 15000 20000 25000 29000

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

all: $(TMP)/vector.tmp $(TMP)/matmul.tmp
	@echo "\033[38;5;70m\nData has been created!\nPlot by running 'make plot'\033[0m\n"

# Reglas de compilación para GPU
$(GPU_DIR)/matmul.x: $(GPU_DIR)/mat_mul.cu | $(TMP)
ifneq ($(filter gpu both,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CUDA) $< -o $@
endif

$(GPU_DIR)/vector.x: $(GPU_DIR)/vector.cu | $(TMP)
ifneq ($(filter gpu both,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CUDA) $< -o $@
endif

# Reglas de compilación para CPU
$(CPU_DIR)/matmul.x: $(CPU_DIR)/mat_mul.cpp | $(TMP)
ifneq ($(filter cpu both,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -o $@
endif

$(CPU_DIR)/vector.x: $(CPU_DIR)/vector.cpp | $(TMP)
ifneq ($(filter cpu both,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -o $@
endif

# Reglas para generar archivos temporales
$(TMP)/vector.tmp: $(GPU_DIR)/vector.x $(CPU_DIR)/vector.x | $(TMP)
ifneq ($(filter gpu both,$(TYPE)),)
	@$(MAKE) vector-times
endif
# ifneq ($(filter cpu both,$(TYPE)),)
# 	@$(MAKE) vector-times
# endif
	@touch $@

$(TMP)/matmul.tmp: $(GPU_DIR)/matmul.x $(CPU_DIR)/matmul.x | $(TMP)
ifneq ($(filter gpu both,$(TYPE)),)
	@$(MAKE) matmul-times
endif
# ifneq ($(filter cpu both,$(TYPE)),)
# 	@$(MAKE) matmul-times
# endif
	@touch $@

vector-times:
ifneq ($(filter gpu both,$(TYPE)),)
	@echo "     \033[1;38;5;214mVECTOR (GPU)\033[0m\n$(header)" && \
	for size in $(SIZES); do ./$(GPU_DIR)/vector.x $$size 2>$(LOG)/gpu_$@.log; done | tee $(DAT)/gpu_$@.txt
endif
ifneq ($(filter cpu both,$(TYPE)),)
	@echo "     \033[1;38;5;214mVECTOR (CPU)\033[0m\n$(header)" && \
	for size in $(SIZES); do ./$(CPU_DIR)/vector.x $$size 2>$(LOG)/cpu_$@.log; done | tee $(DAT)/cpu_$@.txt
endif

# matmul-times:
# ifneq ($(filter gpu both,$(TYPE)),)
# 	@echo "     \033[1;38;5;214mMATMUL (GPU)\033[0m\n$(header)" && \
# 	for size in $(SIZES); do ./$(GPU_DIR)/matmul.x $$size 2>$(LOG)/gpu_$@.log; done | tee $(DAT)/gpu_$@.txt
# endif
# ifneq ($(filter cpu both,$(TYPE)),)
# 	@echo "     \033[1;38;5;214mMATMUL (CPU)\033[0m\n$(header)" && \
# 	for size in $(SIZES); do ./$(CPU_DIR)/matmul.x $$size 2>$(LOG)/cpu_$@.log; done | tee $(DAT)/cpu_$@.txt
# endif