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

# Verificar y crear directorios necesarios
prepare:
	@mkdir -p $(LOG) $(DAT) $(TMP) $(REP)

all: $(TMP)/vector.tmp $(TMP)/matmul.tmp
	@echo "\033[38;5;70m\nData has been created!\nPlot by running 'make plot'\033[0m\n"

# GPU compilation
$(GPU_DIR)/matmul.x: $(GPU_DIR)/mat_mul.cu
ifneq ($(filter gpu both,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CUDA) $< -o $@
endif

$(GPU_DIR)/vector.x: $(GPU_DIR)/vector.cu
ifneq ($(filter gpu both,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CUDA) $< -o $@
endif

# CPU compilation
$(CPU_DIR)/matmul.x: $(CPU_DIR)/mat_mul.cpp
ifneq ($(filter cpu both,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -o $@
endif

$(CPU_DIR)/vector.x: $(CPU_DIR)/vector.cpp
ifneq ($(filter cpu both,$(TYPE)),)
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -o $@
endif


# vector-times: vector.x
# 	@echo "     \033[1;38;5;214mVECTOR\033[0m\n$(header)" && for size in $(SIZES); do ./$< $$size 2>$(LOG)/$@.log; done | tee $(DAT)/$@.txt

# $(TMP)/vector.tmp: vector.x
# 	@$(MAKE) -s vector-times
# 	@touch $@

# matmul-times: matmul.x
# 	@echo "     \033[1;38;5;214mMATMUL\033[0m\n$(header)" && for size in $(SIZES); do ./$< $$size 2>$(LOG)/$@.log; done | tee $(DAT)/$@.txt

# $(TMP)/matmul.tmp: matmul.x
# 	@$(MAKE) -s matmul-times
# 	@touch $@

# plot: plot.py
# 	python3 $<
# 	@echo "-------------------------------------------------"
# 	@echo "Figures are saved to 'figs' directory.          |"
# 	@echo "-------------------------------------------------"

# report:
# 	pdflatex --interaction=batchmode -output-directory=$(REP)/ Entrega-1_PF-HPC-G3.tex

# execs-clean:
# 	rm *.x

# clean:
# 	rm figs/* $(DAT)/* $(LOG)/* $(TMP)/* $(REP)/* *.x
