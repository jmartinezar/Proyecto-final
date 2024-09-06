#!/usr/bin/bash

WSSIZESV=(10000 2 20000 4 40000 8 80000 16 160000 32)

echo -e 'nthreads\tsize\tTime' | tee data/raw/weak-vector-times-gpu.txt

for(( ii=0; ii<${#WSSIZESV[@]}; ii+=2 ));
do
    echo -n ${WSSIZESV[$ii+1]} && echo -e -n '\t' && ./GPU/vector.x ${WSSIZESV[$ii]} ${WSSIZESV[$ii+1]} 2>logs/weak.log;
done | tee data/raw/weak-vector-times-gpu.txt
