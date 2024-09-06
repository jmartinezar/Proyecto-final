#!/usr/bin/bash

WSSIZESM=(1000 2 2000 4 4000 8 8000 16 16000 32 32000 64)

echo -e 'nthreads\tsize\tTime' | tee data/raw/weak-matmul-times-cpu.txt

for(( ii=0; ii<${#WSSIZESM[@]}; ii+=2 ));
do
    echo -n ${WSSIZESM[$ii+1]} && echo -n -e '\t' && OMP_NUM_THREADS=${WSSIZESM[$ii+1]} ./CPU/matmul.x ${WSSIZESM[$ii]} 2>logs/weak_mat_cpu.log;
done | tee data/raw/weak-matmul-times-cpu.txt
