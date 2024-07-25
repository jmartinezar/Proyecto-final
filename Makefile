COMPILER=nvcc
DAT=data
LOG=logs
SIZES=10 100 500 1000 5000 10000 15000 20000 25000 29000

header=size\ttime(s)

all: data

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

# el comando no se incluye en el target all porque el server remoto original de trabajo
# no tiene matplotlib instalado. Se deja como opción para graficar en una máquina con la libreria
plot: plot.py data
	python $<

clean:
	rm data/* logs/* *.x
