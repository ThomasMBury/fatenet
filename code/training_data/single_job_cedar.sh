#!/bin/bash -l
#SBATCH --job-name=gen_training_data
#SBATCH --account=def-glass # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-10:00:00         # adjust this to match the walltime of your job
#SBATCH --nodes=1
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
module load gcc/9.3.0 arrow python scipy-stack

# Create virtual env
echo Create virtual environemnt
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# Install packages
echo Install packages
pip install --no-index --upgrade pip

pip install numpy
pip install pandas

# Begin python job (unbuffered stdout)
echo Begin python job
python -u gen_training_data_cont.py --nsims 100 --ncells 20 --path_out "output_80k/" --seed $SEED
