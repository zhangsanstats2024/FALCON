#!/bin/bash 
#SBATCH --partition=xeon-g6-volta 
#SBATCH --constraint=xeon-g6
#SBATCH -t 0-96:0 #Request runtime of 30 minutes
#SBATCH -o ../logs/output_run_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e ../logs/error_run_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --gres=gpu:volta:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --array=0


module purge
module load anaconda/2022b
module load /home/gridsan/groups/datasets/ImageNet/modulefile

TASK_ID=$SLURM_ARRAY_TASK_ID
EXP_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"


echo $TASK_ID

echo $EXP_ID

algos=("MP" "WF" "BS" "IHT" "WFbeta")
algo=${algos[2]}

block_sizes=(2000 -1)
split_types=(1 -1)
block_size=${block_sizes[0]}
split_type=${split_types[0]}

nums_stages=(1 16 16)
num_stages=${nums_stages[0]}



sparsity_schedule="poly"

training_schedules=("cosine_fast_works_098" "cosine_fast1" "cosine_one")
training_schedule=${training_schedules[0]}

CHECKPOINT_PATH="/home/gridsan/rbenbaki/NetworkPruningFlops/NetworkPruning/results/mobilenetv1_imagenet_grad_90_95/data22335035_0_1681156954.csv_epoch47.pth"
FIRST_EPOCH=48


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

fisher_subsample_sizes=(1000)
fisher_subsample_size=${fisher_subsample_sizes[0]}
fisher_mini_bszs=(16)
fisher_mini_bsz=16

l2s=(0.0001 0.001)
l2=0.01


### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$((12302 + TASK_ID))
export WORLD_SIZE=4
echo $MASTER_PORT

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python3 -u run_expflop_gradual.py --arch mobilenetv1 --dset imagenet  --num_workers 20 --exp_name grad_90_9297 --exp_id ${EXP_ID} \
--compute_training_losses False --restrict_support True --shuffle_train True --use_wbar True --use_activeset True \
--test_batch_size 256 --train_batch_size 256  --fisher_subsample_size ${fisher_subsample_size} --fisher_mini_bsz ${fisher_mini_bsz} --fisher_data_bsz ${fisher_subsample_size} \
--num_iterations 500 --num_stages ${num_stages} --seed ${seed} --first_order_term False --compute_trace_H False --recompute_X True \
--sparsity 0.9 --base_level 0.3 --outer_base_level 0.5  --flop_ratio 0.9297 --flop_base 0.3 --l2 ${l2}  \
--sparsity_schedule ${sparsity_schedule} --training_schedule ${training_schedule} --algo ${algo} --normalize False --block_size 2000 \
--split_type ${split_type} --max_lr ${max_lr} --min_lr ${min_lr} --prune_every ${prune_every} --nprune_epochs ${nprune_epochs} --nepochs ${nepochs} \
--gamma_ft ${gamma_ft} --warm_up ${warm_up} --ft_max_lr ${ft_max_lr} --ft_min_lr ${ft_min_lr} --checkpoint_path ${CHECKPOINT_PATH} --first_epoch ${FIRST_EPOCH}
