#!/bin/bash -l
#SBATCH --job-name=train_dl_model
#SBATCH --account=def-glass # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-2:00:00         # adjust this to match the walltime of your job
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4      # adjust this if you are using parallel commands
#SBATCH --mem=32000M             # adjust this according to the memory requirement per node you need
#SBATCH --mail-user=thomas.bury@mcgill.ca # adjust this to match your email address
#SBATCH --mail-type=END
#SBATCH --output=stdout/job-%j.out

echo Job $SLURM_JOB_ID released

# Load modules
echo Load modules
module load StdEnv/2020
module load python/3.10
module load gcc/9.3.0 arrow python scipy-stack cuda cudnn

# Create virtual env
echo Create virtual environemnt
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# Install packages
echo Install packages
pip install --no-index --upgrade pip

pip install tensorflow==2.10
pip install scikit-learn
pip install arch==4.19
pip install ewstools
pip install matplotlib
pip install seaborn
pip install kaleido
pip install wandb
pip install tyro

# Use offline mmode and sync later (see https://docs.alliancecan.ca/wiki/Weights_%26_Biases_(wandb))
# wandb offline


# Begin python job (unbuffered stdout)
echo Begin python job
python -u dl_train.py --num_epochs 200 --wandb_project_name "FateNet_ensemble_orig" --path_training "../training_data/output_80k/" --detrend --batch_size 1024 --model_type $MODEL_TYPE --num_conv_layers $NUM_CONV_LAYERS --num_conv_filters $NUM_CONV_FILTERS --mem_cells_1 $MEM_CELLS_1 --mem_cells_2 $MEM_CELLS_2
