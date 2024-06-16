#!/bin/bash

# Parameter values to loop through
declare -a SEED_VALS=(0 1 2 3 4 5 6 7 8 9);

for SEED in "${SEED_VALS[@]}"; do
    echo "Running job for seed=$SEED"
    
    # Put variables in environment
    export SEED=$SEED
    
    # Run job on cedar
    sbatch single_job_cedar.sh
    sleep 1.0
	
done

