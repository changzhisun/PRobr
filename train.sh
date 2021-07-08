data_dir=./data/depth-5
lr=1e-5
ep=50
output_dir=./output/d5
# model=roberta_rr # PRover
model=roberta_probr  # PRobr
num_gpus=8
distribute_on=" -m torch.distributed.launch --nproc_per_node=$num_gpus"
distribute_off=""
distribute=$distribute_off

python3 $distribute ./run_experiment_probr.py                        \
    --data_dir $data_dir                                             \
    --output_dir $output_dir                                         \
    --per_gpu_eval_batch_size 32                                      \
    --per_gpu_train_batch_size 8                                    \
    --model_type $model                                   \
    --model_name_or_path roberta-large                               \
    --task_name rr                                                   \
    --do_train                                                       \
    --do_eval                                                        \
    --do_lower_case                                                  \
    --max_seq_length 300                                             \
    --learning_rate $lr                                              \
    --gradient_accumulation_steps 1                                  \
    --num_train_epochs $ep                                             \
    --logging_steps 4752                                             \
    --save_steps 4750                                                \
    --seed 42                                                        \
    --data_cache_dir ./output/cache/                                 \
    --evaluate_during_training                                       \
