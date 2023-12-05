#!/bin/bash 
#SBATCH -c 20
#SBATCH -t 0-72:0 #Request runtime of 30 minutes
#SBATCH --gres=gpu:volta:1
#SBATCH --exclusive
#SBATCH -o ../logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --array=0-0


sleep $SLURM_ARRAY_TASK_ID

module purge
module load anaconda/2022b
module load /home/gridsan/groups/datasets/ImageNet/modulefile

TASK_ID=$SLURM_ARRAY_TASK_ID
echo $TASK_ID


algos=("BS")
seed=("1")
sparsitys=(-1)
l2=("0.001")
flop_ratio=0.6
num_stages=1
block_size=2000
sparsity_schedule="poly"

sparsity=0.6
algo="BS"

#algo=${algos[$(($TASK_ID % 2))]}
#TASK_ID=$((TASK_ID/2))

#sparsity=${sparsitys[$(($TASK_ID % 2))]}
#TASK_ID=$((TASK_ID/2))



python3 -u run_expflop.py --arch resnet50down --dset imagenet --num_workers 40 --exp_name flop_single_test01 --exp_id ${SLURM_ARRAY_TASK_ID} --compute_training_losses False --restrict_support True --shuffle_train True --use_wbar True --use_activeset True --test_batch_size 400 --fisher_subsample_size 500 --fisher_mini_bsz 1 --fisher_data_bsz 300 --num_iterations 500 --num_stages ${num_stages} --seed ${seed} --first_order_term False --compute_trace_H False --recompute_X True --sparsity ${sparsity} --base_level 0.3 --flop_ratio ${flop_ratio} --flop_base 0.3 --l2 ${l2} --sparsity_schedule "poly" --algo ${algo} --normalize False --block_size ${block_size} --split_type -1 


