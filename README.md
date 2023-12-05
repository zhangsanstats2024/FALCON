# FALCON

### Requirement: 
+ Python 3.8
+ Pytorch 1.10

### running 
+ one-shot pruning on resnet50: python3 -u run_expflop.py --arch resnet50down --dset imagenet --num_workers 40 --exp_name flop_single_test01 --exp_id 1 \
--compute_training_losses False --restrict_support True --shuffle_train True --use_wbar True --use_activeset True --test_batch_size 400 \
--fisher_subsample_size 500 --fisher_mini_bsz 1 --fisher_data_bsz 300 --num_iterations 500 --num_stages 20 --seed 1 \
--first_order_term False --compute_trace_H False --recompute_X True --sparsity 0.3 --flop_ratio 0.3 --l2 0.0001 \
--sparsity_schedule "poly" --algo "BS" --normalize False --block_size 2000 --split_type -1 
+ gradual pruning on resnet50: srun python3 -u run_expflop_gradual.py --arch mobilenetv1 --dset imagenet  --num_workers 20 --exp_name 1 --exp_id 1 \
--compute_training_losses False --restrict_support True --shuffle_train True --use_wbar True --use_activeset True \
--test_batch_size 256 --train_batch_size 256  --fisher_subsample_size 1000 --fisher_mini_bsz 1000 --fisher_data_bsz 1000 \
--num_iterations 500 --num_stages 12 --seed 1 --first_order_term False --compute_trace_H False --recompute_X True \
--sparsity 0.9 --base_level 0.3 --outer_base_level 0.5  --flop_ratio 0.9297 --flop_base 0.3 --l2 0.01  \
--sparsity_schedule "poly" --training_schedule "cosine_fast_works_098" --algo "BS" --normalize False --block_size 2000 \
--split_type -1 --max_lr 0.1 --min_lr 0.00001 --prune_every 12 --nprune_epochs 7 --nepochs 100 
  
