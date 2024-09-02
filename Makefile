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

# Verificar y crear directorios necesarios
prepare:
	@mkdir -p $(LOG) $(DAT) $(TMP) $(REP)

all: $(TMP)/vector.tmp $(TMP)/matmul.tmp
	@echo "\033[38;5;70m\nData has been created!\nPlot by running 'make plot'\033[0m\n"

	ifeq ($(TYPE), cpu)
	matmul_cpu.x: CPU/mat_mul.cpp
		@echo "\n\033[1;38;5mCompiling $< \033[0m"
		$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -o $@

	vector_cpu.x: CPU/vector.cpp
		@echo "\n\033[1;38;5mCompiling $< \033[0m"
		$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -o $@
	else ifeq ($(TYPE), gpu)
	matmul_gpu.x: GPU/mat_mul.cu
		@echo "\n\033[1;38;5mCompiling $< \033[0m"
		$(COMPILER_CUDA) $< -o $@

	vector_gpu.x: GPU/vector.cu
		@echo "\n\033[1;38;5mCompiling $< \033[0m"
		$(COMPILER_CUDA) $< -o $@

	else ifeq ($(TYPE), both)
	matmul_cpu.x: CPU/mat_mul.cpp
		@echo "\n\033[1;38;5mCompiling $< \033[0m"
		$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -o $@

	matmul_gpu.x: GPU/mat_mul.cu
		@echo "\n\033[1;38;5mCompiling $< \033[0m"
		$(COMPILER_CUDA) $< -o $@

	vector_cpu.x: CPU/vector.cpp
		@echo "\n\033[1;38;5mCompiling $< \033[0m"
		$(COMPILER_CPP) $(CXXFLAGS_CPP) $< -o $@

	vector_gpu.x: GPU/vector.cu
		@echo "\n\033[1;38;5mCompiling $< \033[0m"
		$(COMPILER_CUDA) $< -o $@
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
