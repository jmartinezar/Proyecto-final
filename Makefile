COMPILER=nvcc
DAT=data
LOG=logs
SIZES=10 100 500 1000 5000 10000 15000 20000 25000 29000

header=size\ttime(s)

all: plot

matmul.x: mat_mul.cu
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER) $< -o $@

vector.x: vector.cu
	@echo "\n\033[1;38;5mCompiling $< \033[0m"
	$(COMPILER) $< -o $@

matmul: matmul.x
	./$<

vector: vector.x
	./$<

vector-times: vector.x
	@echo "     \033[1;38;5;214mVECTOR\033[0m\n$(header)" && for size in $(SIZES); do ./$< $$size 2>$(LOG)/$@.log; done | tee $(DAT)/$@.txt

matmul-times: matmul.x
	@echo "     \033[1;38;5;214mMATMUL\033[0m\n$(header)" && for size in $(SIZES); do ./$< $$size 2>$(LOG)/$@.log; done | tee $(DAT)/$@.txt

data: vector-times matmul-times

plot: plot.py data
	python3 $<
	@echo "Figures should be created in figs/, if not, check logs from python3 execution"

execs-clean:
	rm *.x

clean:
	rm figs/*.pdf data/*.txt logs/*.log *.x
