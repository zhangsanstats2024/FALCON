#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 0-48:0 #Request runtime of 30 minutes
#SBATCH --gres=gpu:volta:1
#SBATCH --exclusive
#SBATCH -o ../logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-0

module load anaconda/2022b

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID


#num_iterations num_stages sparsity flop_ratio l2 algo
#first_order_term -> use if multistage
#

algos=("WFbeta")
algo=${algos[0]}

flop_ratios=(0.8)
flop_ratio=${flop_ratios[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

sparsitys=(0.55)
sparsity=${sparsitys[$(($TASK_ID % 1))]}
TASK_ID=$((TASK_ID/1))

num_stages=1
block_size=2000


seed=("1")
l2=("1 10")



sparsity_schedule="poly"
fisher_subsample_size=1000



python3 -u run_expflop.py --arch resnet20 --dset cifar10 --num_workers 40 --exp_name flop_multi_test9 --exp_id ${SLURM_ARRAY_TASK_ID} --compute_training_losses False --restrict_support True --shuffle_train True --use_wbar True --use_activeset True --test_batch_size 400 --fisher_subsample_size ${fisher_subsample_size} --fisher_mini_bsz 1 --fisher_data_bsz 300 --num_iterations 500 --num_stages ${num_stages} --seed ${seed} --first_order_term False --compute_trace_H False --recompute_X True --sparsity ${sparsity} --base_level 0.3 --flop_ratio ${flop_ratio} --flop_base 0.3 --l2 ${l2} --sparsity_schedule "poly" --algo ${algo} --normalize False --block_size ${block_size} --split_type -1 