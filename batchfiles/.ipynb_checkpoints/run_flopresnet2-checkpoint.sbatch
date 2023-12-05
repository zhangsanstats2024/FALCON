#!/bin/bash 
#SBATCH --partition=xeon-g6-volta 
#SBATCH --constraint=xeon-g6
#SBATCH -t 0-12:0 #Request runtime of 30 minutes
#SBATCH -o ../logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --array=0


module purge
module load anaconda/2022b
module load /home/gridsan/groups/datasets/ImageNet/modulefile

TASK_ID=$SLURM_ARRAY_TASK_ID
EXP_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"



echo $TASK_ID

echo $EXP_ID

algos=("Heuristic_LSBlock" "MP")
block_sizes=(500 -1)
split_types=(1 -1)
algo=${algos[0]}
block_size=${block_sizes[0]}
split_type=${split_types[0]}

nums_stages=(1 16 16)
num_stages=${nums_stages[0]}



sparsity_schedule="poly"

training_schedules=("cosine_fast_works_098" "cosine_fast1" "cosine_one")
training_schedule=${training_schedules[0]}



if [ $training_schedule == "cosine_fast_works_098" ] 
then 
    max_lr=0.1
    min_lr=0.00001
    prune_every=12
    nprune_epochs=7
    nepochs=100
    warm_up=0
    ft_max_lr=0.1
    ft_min_lr=0.00001
    gamma_ft=-1
fi

echo $max_lr

seed=2

fisher_subsample_sizes=(500)
fisher_subsample_size=${fisher_subsample_sizes[0]}

l2s=(0.0001 0.001)
l2=${l2s[0]}

fisher_mini_bszs=(16)
fisher_mini_bsz=16

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$((12390 + TASK_ID))
export WORLD_SIZE=1
echo $MASTER_PORT

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python3 -u run_expflop_gradual.py --arch resnet20 --dset cifar10  --num_workers 20 --exp_name grad_mar14 --exp_id 1 --compute_training_losses False --restrict_support True --shuffle_train True --use_wbar True --use_activeset True --test_batch_size 256 --train_batch_size 256  --fisher_subsample_size 1000 --fisher_mini_bsz 1 --fisher_data_bsz 300 --num_iterations 500 --num_stages 1 --seed 0 --first_order_term False --compute_trace_H False --recompute_X True --sparsity 0.7 --base_level 0.3 --outer_base_level 0.5  --flop_ratio 0.7 --flop_base 0.3 --l2 0.01 --sparsity_schedule "poly" --training_schedule "cosine_fast_works_098" --algo BS --normalize False --block_size 2000  --split_type -1 --max_lr 0.1 --min_lr 0.00001 --prune_every 1 --nprune_epochs 5 --nepochs 5 --gamma_ft -1 --warm_up 0 --ft_max_lr 0.1 --ft_min_lr 0.00001
