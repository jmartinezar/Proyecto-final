all: mm
	./$<

mm: mat_mul.cu
	nvcc $< -o $@

vec: vector
	./$<

vector: vector.cu
	nvcc $< -o $@
