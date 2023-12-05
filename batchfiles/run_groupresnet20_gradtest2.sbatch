#!/bin/bash 
#SBATCH --partition=xeon-g6-volta 
#SBATCH --constraint=xeon-g6
#SBATCH -t 0-12:0 #Request runtime of 30 minutes
#SBATCH -o ../logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --array=0-0


sleep $SLURM_ARRAY_TASK_ID

module purge
module load anaconda/2022b
module load /home/gridsan/groups/datasets/ImageNet/modulefile

TASK_ID=$SLURM_ARRAY_TASK_ID
EXP_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"


echo $TASK_ID

echo $EXP_ID


srun python3 -u run_expgroup_gradual.py --arch resnet20 --dset cifar10  --num_workers 20 --exp_name group_gradualtest --exp_id 0 --compute_training_losses False --restrict_support True --shuffle_train True --use_wbar True --use_activeset True --test_batch_size 256 --train_batch_size 64  --fisher_subsample_size 1000 --fisher_mini_bsz 1 --fisher_data_bsz 1000 --num_iterations 500 --num_stages 1 --seed 1 --first_order_term False --compute_trace_H False --recompute_X True --sparsity 0.5 --base_level 0.3 --outer_base_level 0.5 --l2 0.01  --sparsity_schedule poly --training_schedule cosine_fast_works_098 --algo BS --normalize False --block_size 2000 --split_type -1 --max_lr 0.01 --min_lr 0.00001 --prune_every 1 --nprune_epochs 2 --nepochs 3 --gamma_ft -1 --warm_up 0 --ft_max_lr 0.01 --ft_min_lr 0.00001 --prune_type overlap