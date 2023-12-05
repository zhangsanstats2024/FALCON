#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 0-48:0 #Request runtime of 30 minutes
#SBATCH --gres=gpu:volta:1
#SBATCH -o ../logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-0

module purge
module load anaconda/2022b
module load /home/gridsan/groups/datasets/ImageNet/modulefile

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID


sparsity=0.7
flop_ratio=0.7

algos=("MP" "WF" "BS" "IHT")
algo="BS"

num_stages=1

seeds=(1 2 3 4)
seed=1

first_order_term=True
block_size=2000
sparsity_schedule="poly"
num_iterations=500
fisher_subsample_size=1000
l2=("0.0001 0.001 0.01 0.1 1 10")


python3 -u run_expflop.py --arch mobilenetv1 --dset imagenet --num_workers 40 --exp_name flop_multi_xiang --exp_id 1 --compute_training_losses False --restrict_support True --shuffle_train True --use_wbar True --use_activeset True --test_batch_size 400 --fisher_subsample_size ${fisher_subsample_size} --fisher_mini_bsz 1 --fisher_data_bsz 300 --num_iterations ${num_iterations} --num_stages ${num_stages} --seed ${seed} --first_order_term ${first_order_term} --compute_trace_H False --recompute_X True --sparsity ${sparsity} --base_level 0.2 --flop_ratio ${flop_ratio} --flop_base 0.2 --l2 ${l2} --sparsity_schedule ${sparsity_schedule} --algo ${algo} --normalize False --block_size ${block_size} --split_type -1 
