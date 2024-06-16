#!/bin/bash

# Parameter values to loop through
declare -a SEED_VALS=(1 2 3 4 5);

for SEED in "${SEED_VALS[@]}"; do

    echo "Running job for SEED=$SEED"

    # export env variables for model type 1
    export MODEL_TYPE=1
    export SEED=$SEED
    export NUM_CONV_LAYERS=1
    export NUM_CONV_FILTERS=50
    export MEM_CELLS_1=50
    export MEM_CELLS_2=10

    # Run job on cedar
    sbatch single_job_cedar.sh
    sleep 1.0

    # export env variables for model type 1
    export MODEL_TYPE=2
    export SEED=$SEED
    export NUM_CONV_LAYERS=1
    export NUM_CONV_FILTERS=50
    export MEM_CELLS_1=50
    export MEM_CELLS_2=10
    
    sbatch single_job_cedar.sh
    sleep 1.0
   
done





