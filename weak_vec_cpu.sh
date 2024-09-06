#!/usr/bin/bash

WSSIZESV=(10000 2 20000 4 40000 8 80000 16 160000 32 320000 64)

echo -e 'nthreads\tsize\tTime' | tee data/raw/weak-vector-times-cpu.txt

for(( ii=0; ii<${#WSSIZESV[@]}; ii+=2 ));
do
    echo -n ${WSSIZESV[$ii+1]} && echo -n -e '\t' && OMP_NUM_THREADS=${WSSIZESV[$ii+1]} ./CPU/vector.x ${WSSIZESV[$ii]} 2>logs/weak_vec_cpu.log;
done | tee data/raw/weak-vector-times-cpu.txt
