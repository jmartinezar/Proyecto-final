#!/usr/bin/bash

WSSIZESM=(1000 2 1260 4 1587 8 2000 16 2520 32)

echo -e 'nthreads\tsize\tTime' | tee data/raw/weak-matmul-times-gpu.txt

for(( ii=0; ii<${#WSSIZESM[@]}; ii+=2 ));
do
    echo -n -e ${WSSIZESM[$ii+1]} && echo -n -e '\t' && ./GPU/matmul.x ${WSSIZESM[$ii]} ${WSSIZESM[$ii+1]} 2>logs/weak_mat.log;
done | tee data/raw/weak-matmul-times-gpu.txt
