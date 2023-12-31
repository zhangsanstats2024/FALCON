#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 0-72:0 #Request runtime of 30 minutes
#SBATCH --gres=gpu:volta:1
#SBATCH --exclusive
#SBATCH -o ../logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-2



module load anaconda/2022b

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID


seed=("1 2 3 4 5")
l2=("0.0001 0.001 0.01")

flop_ratio=0.4
sparsity=0.4
num_stages=20
block_size=2000
sparsity_schedule="poly"


algo="BS"

alphas=(0.01 0.001 0.0001)
alpha_scale=${alphas[$(($TASK_ID % 3))]}
TASK_ID=$((TASK_ID/3))

python3 -u run_expflop.py --arch resnet20 --dset cifar10 --num_workers 40 --exp_name flop_multi_flop04_bs_Apr29 --exp_id ${SLURM_ARRAY_TASK_ID} --compute_training_losses False --restrict_support True --shuffle_train True --use_wbar True --use_activeset True --test_batch_size 400 --fisher_subsample_size 1000 --fisher_mini_bsz 1 --fisher_data_bsz 300 --num_iterations 500 --num_stages ${num_stages} --seed ${seed} --first_order_term True --compute_trace_H False --recompute_X True --sparsity ${sparsity} --base_level 0.2 --flop_ratio ${flop_ratio} --flop_base 0.2 --l2 ${l2} --sparsity_schedule "poly" --algo ${algo} --normalize False --block_size ${block_size} --split_type -1 --alpha_scale ${alpha_scale}