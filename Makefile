COMPILER=nvcc
DAT=data
LOG=logs
TMP=tmp
SIZES=10 100 500 1000 5000 10000 15000 20000 25000 29000

header=size\ttime(s)

all: $(TMP)/vector.tmp $(TMP)/matmul.tmp
	@echo "\033[38;5;70m\nData has been created!\nPlot by running 'make plot'\033[0m\n"

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

$(TMP)/vector.tmp: vector.x
	@$(MAKE) -s vector-times
	@touch $@

matmul-times: matmul.x
	@echo "     \033[1;38;5;214mMATMUL\033[0m\n$(header)" && for size in $(SIZES); do ./$< $$size 2>$(LOG)/$@.log; done | tee $(DAT)/$@.txt

$(TMP)/matmul.tmp: matmul.x
	@$(MAKE) -s matmul-times
	@touch $@

plot: plot.py 
	python3 $<
	@echo "-------------------------------------------------"
	@echo "Figures are saved to 'figs' directory.          |\nIf not, check dependencies and logs in terminal.|"
	@echo "-------------------------------------------------"

execs-clean:
	rm *.x

clean:
	rm figs/*.pdf $/*.txt logs/*.log  *.x
