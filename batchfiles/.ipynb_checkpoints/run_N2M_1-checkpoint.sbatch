#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 0-48:0 #Request runtime of 30 minutes
#SBATCH --gres=gpu:volta:1
#SBATCH --exclusive
#SBATCH -o ../logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-111

module load anaconda/2022b

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID

python N2M_multi.py $SLURM_ARRAY_TASK_ID